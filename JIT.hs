{-# LANGUAGE OverloadedStrings #-}

module JIT (jitPass) where

import LLVM.AST hiding (Type, Add, Mul, Sub)
import qualified LLVM.AST as L
import qualified LLVM.AST.Global as L
import qualified LLVM.AST.CallingConvention as L
import qualified LLVM.AST.Type as L
import qualified LLVM.AST.Float as L
import qualified LLVM.AST.Constant as C
import qualified LLVM.AST.IntegerPredicate as L

import Control.Monad
import Control.Monad.Except (throwError)
import Control.Monad.State
import Control.Monad.Writer (tell)
import Control.Applicative (liftA, liftA2)

import Data.Foldable (toList)
import Data.List (intercalate, transpose)
import Data.Traversable
import Data.Functor.Identity

import qualified Foreign.Ptr as F
import Data.ByteString.Short (ShortByteString, toShort, fromShort)
import Data.ByteString.Char8 (pack, unpack)
import Data.Word (Word64 (..))

import Type
import Syntax
import Env
import Record
import Util
import Imp
import Pass
import Fresh
import PPrint

import LLVMExec

-- TODO: figure out whether we actually need type everywhere here
data Ptr w = Ptr w L.Type  deriving (Show)

data JitVal w = ScalarVal w L.Type
              | ArrayVal (Ptr w) [w]  deriving (Show)


data Cell = Cell (Ptr Operand) [Operand]
type CompileVal  = JitVal Operand
type PersistVal  = JitVal Word64
type PersistEnv = Env PersistVal
type ImpVarEnv = Env (Either CompileVal Cell)

data CompileState = CompileState { curBlocks   :: [BasicBlock]
                                 , curInstrs   :: [NInstr]
                                 , scalarDecls :: [NInstr]
                                 , blockName :: L.Name
                                 , impVarEnv  :: ImpVarEnv
                                 }

type CompileM a = MonadPass () CompileState a
data CompiledProg = CompiledProg [CompileVal] Module
data ExternFunSpec = ExternFunSpec ShortByteString L.Type [L.Type] [ShortByteString]

type Long = Operand
type NInstr = Named Instruction

jitPass :: Pass PersistEnv ImpDecl ()
jitPass decl = case decl of
  ImpTopLet bs prog -> do vals <- evalProg prog
                          put $ newEnv $ zip (map fst bs) vals
  ImpEvalCmd _ NoOp -> return ()
  ImpEvalCmd ty (Command cmd prog) -> case cmd of
    Passes -> do CompiledProg vs m <- toLLVM prog
                 llvm <- liftIO $ showLLVM m
                 tell ["\n\nLLVM\n" ++ llvm]
    EvalExpr -> do vals <- evalProg prog
                   vecs <- liftIO $ mapM asVec vals
                   tell [pprint (restructureVal ty vecs)]
    _ -> return ()

evalProg :: ImpProgram -> TopMonadPass PersistEnv [PersistVal]
evalProg prog = do
  CompiledProg outvals mod <- toLLVM prog
  outWords <- liftIO $ evalJit (length (JitVals outvals)) mod
  return $ asPersistVals outWords outvals

toLLVM :: ImpProgram -> TopMonadPass PersistEnv CompiledProg
toLLVM prog = do
  env <- gets $ fmap (Left . asCompileVal)
  let initState = CompileState [] [] [] "start_block" env
  liftExcept $ evalPass () initState nameRoot (compileProg prog)

asCompileVal :: PersistVal -> CompileVal
asCompileVal (ScalarVal word ty) = ScalarVal (constOperand (baseTy ty) word) ty
asCompileVal (ArrayVal (Ptr ptr ty) shape) = ArrayVal (Ptr ptr' ty) shape'
  where ptr' = L.ConstantOperand $ C.IntToPtr (C.Int 64 (fromIntegral ptr)) (L.ptr ty)
        shape' = map (constOperand IntType) shape

asPersistVals :: [Word64] -> [CompileVal] -> [PersistVal]
asPersistVals words vals = case restructure words (JitVals vals) of
                                  JitVals vals' -> vals'

-- TODO: concretize type with actual index set
restructureVal :: Type -> [Vec] -> Value
restructureVal ty vecs = Value ty $ restructure vecs (typeLeaves ty)
  where
    typeLeaves :: Type -> RecTree ()
    typeLeaves ty = case ty of BaseType b -> RecLeaf ()
                               TabType _ valTy -> typeLeaves valTy
                               RecType r -> RecTree $ fmap typeLeaves r
                               _ -> error $ "can't show " ++ pprint ty

asVec :: PersistVal -> IO Vec
asVec v = case v of
  ScalarVal x ty ->  return $ cast (baseTy ty) [x]
  ArrayVal (Ptr ptr ty) shape -> do let size = fromIntegral $ foldr (*) 1 shape
                                    words <- readPtrs size (wordAsPtr ptr)
                                    return $ cast (baseTy ty) words
  where cast IntType xs = IntVec $ map fromIntegral xs

constOperand :: BaseType -> Word64 -> Operand
constOperand IntType  x = litInt (fromIntegral x)
constOperand RealType x = error "floating point not yet implemented"

compileProg :: ImpProgram -> CompileM CompiledProg
compileProg (ImpProgram statements outExprs) = do
  mapM compileStatement statements
  vals <- mapM compileExpr outExprs
  finalReturn vals
  decls <- gets scalarDecls
  blocks <- gets (reverse . curBlocks)
  return $ CompiledProg vals (makeModule decls blocks)

compileStatement :: Statement -> CompileM ()
compileStatement statement = case statement of
  Update v idxs expr -> do val <- compileExpr expr
                           cell <- lookupCellVar v
                           idxs' <- mapM lookupValVar idxs
                           cell' <- idxCell cell idxs'
                           writeCell cell' val
  ImpLet (v, _) expr -> do val <- compileExpr expr
                           modify $ setImpVarEnv (addV (v, (Left val)))
  Alloc v (IType b shape) -> do
    shape' <- mapM lookupValVar shape
    cell <- case shape' of [] -> alloca b (topTag v)
                           _ -> malloc b shape' (topTag v)
    modify $ setImpVarEnv (addV (v, (Right cell)))

  Loop i n body -> do n' <- lookupValVar n
                      compileLoop i n' body
  where unleft (Left x) = x

compileExpr :: IExpr -> CompileM CompileVal
compileExpr expr = case expr of
  ILit v -> return $ ScalarVal (litVal v) (scalarTy (litType v))
  IVar v -> do x <- lookupImpVar v
               case x of
                 Left val -> return val
                 Right (Cell ptr@(Ptr _ ty) shape) -> case shape of
                    [] -> do { op <- load ptr; return $ ScalarVal op ty }
                    _ -> return $ ArrayVal ptr shape
  IGet v i -> do ArrayVal ptr (_:shape) <- compileExpr v
                 ScalarVal i' _ <- lookupValVar i
                 ptr'@(Ptr _ ty) <- indexPtr ptr shape i'
                 case shape of
                   [] -> do x <- load ptr'
                            return $ ScalarVal x ty
                   _  -> return $ ArrayVal ptr' shape

  IBuiltinApp b exprs -> do vals <- mapM compileExpr exprs
                            compileBuiltin b vals

lookupImpVar :: Var -> CompileM (Either CompileVal Cell)
lookupImpVar v = gets $ (! v) . impVarEnv

lookupValVar :: Var -> CompileM CompileVal
lookupValVar v = do { Left val <- lookupImpVar v; return val }

lookupCellVar :: Var -> CompileM Cell
lookupCellVar v = do { Right cell <- lookupImpVar v; return cell }

indexPtr :: Ptr Operand -> [Operand] -> Operand -> CompileM (Ptr Operand)
indexPtr (Ptr ptr ty) shape i = do
  stride <- foldM mul (litInt 1) shape
  n <- mul stride i
  ptr' <- evalInstr "ptr" (L.ptr ty) $ L.GetElementPtr False ptr [n] []
  return (Ptr ptr' ty)

finalReturn :: [CompileVal] -> CompileM ()
finalReturn vals = do
  voidPtr <- evalInstr "" charPtrTy (externCall mallocFun [litInt numBytes])
  outPtr <- evalInstr "out" intPtrTy $ L.BitCast voidPtr intPtrTy []
  foldM writeVal (Ptr outPtr longTy) (JitVals vals)
  finishBlock (L.Ret (Just outPtr) []) (L.Name "")
  where numBytes = 8 * (length (JitVals vals))
        writeVal :: Ptr Operand -> Operand -> CompileM (Ptr Operand)
        writeVal ptr x = store ptr x >> addPtr ptr (litInt 1)


finishBlock :: L.Terminator -> L.Name -> CompileM ()
finishBlock term newName = do
  oldName <- gets blockName
  instrs  <- gets curInstrs
  let newBlock = L.BasicBlock oldName (reverse instrs) (L.Do term)
  modify $ setCurBlocks (newBlock:)
         . setCurInstrs (const [])
         . setBlockName (const newName)

compileLoop :: Var -> CompileVal -> [Statement] -> CompileM ()
compileLoop iVar (ScalarVal n _) body = do
  loopBlock <- freshName "loop"
  nextBlock <- freshName "cont"
  Cell i [] <- alloca IntType "i"
  store i (litInt 0)
  entryCond <- load i >>= (`lessThan` n)
  finishBlock (L.CondBr entryCond loopBlock nextBlock []) loopBlock
  iVal <- load i
  modify $ setImpVarEnv $ addV (iVar, (Left $ ScalarVal iVal longTy)) -- shadows...
  mapM compileStatement body
  iValInc <- add iVal (litInt 1)
  store i iValInc
  loopCond <- iValInc `lessThan` n
  finishBlock (L.CondBr loopCond loopBlock nextBlock []) nextBlock

freshName :: String -> CompileM L.Name
freshName s = do name <- fresh s
                 return $ Name (toShort (pack (pprint name)))

idxCell :: Cell -> [CompileVal] -> CompileM Cell
idxCell cell [] = return cell
idxCell (Cell ptr (_:shape)) (i:idxs) = do
  size <- sizeOf shape
  step <- mul size (scalarOp i)
  ptr' <- addPtr ptr step
  idxCell (Cell ptr' shape) idxs

readCell :: Cell -> CompileM CompileVal
readCell (Cell ptr@(Ptr _ ty) []) = do x <- load ptr
                                       return $ ScalarVal x ty

writeCell :: Cell -> CompileVal -> CompileM ()
writeCell (Cell ptr []) (ScalarVal x _) = store ptr x
writeCell (Cell (Ptr dest _) shape) (ArrayVal (Ptr src _) shape') = do
  numScalars <- sizeOf shape
  numBytes <- mul (litInt 8) numScalars
  addInstr $ L.Do (externCall memcpyFun [dest, src, numBytes])

litVal :: LitVal -> Operand
litVal lit = case lit of
  IntLit  x -> L.ConstantOperand $ C.Int 64 (fromIntegral x)
  RealLit x -> L.ConstantOperand $ C.Float (L.Double x)

litInt :: Int -> Operand
litInt x = L.ConstantOperand $ C.Int 64 (fromIntegral x)

store :: Ptr Operand -> Operand -> CompileM ()
store (Ptr ptr _) x =  addInstr $ L.Do $ L.Store False ptr x Nothing 0 []

load :: Ptr Operand -> CompileM Operand
load (Ptr ptr ty) = evalInstr "" ty $ L.Load False ptr Nothing 0 []

lessThan :: Long -> Long -> CompileM Long
lessThan x y = evalInstr "lt" longTy $ L.ICmp L.SLT x y []

add :: Long -> Long -> CompileM Long
add x y = evalInstr "add" longTy $ L.Add False False x y []

evalInstr :: String -> L.Type -> Instruction -> CompileM Operand
evalInstr s ty instr = do v <- freshName s
                          addInstr $ v L.:= instr
                          return $ L.LocalReference ty v

addPtr :: Ptr Operand -> Long -> CompileM (Ptr Operand)
addPtr (Ptr ptr ty) i = do ptr' <- evalInstr "ptr" (L.ptr ty) instr
                           return $ Ptr ptr' ty
  where instr = L.GetElementPtr False ptr [i] []

alloca :: BaseType -> String -> CompileM Cell
alloca ty s = do v <- freshName s
                 modify $ setScalarDecls ((v L.:= instr):)
                 return $ Cell (Ptr (L.LocalReference (L.ptr ty') v) ty') []
  where ty' = scalarTy ty
        instr = L.Alloca ty' Nothing 0 []

malloc :: BaseType -> [CompileVal] -> String -> CompileM Cell
malloc ty shape s = do
    size <- sizeOf shape'
    n <- mul (litInt 8) size
    voidPtr <- evalInstr "" charPtrTy (externCall mallocFun [n])
    ptr <- evalInstr s (L.ptr ty') $ L.BitCast voidPtr (L.ptr ty') []
    return $ Cell (Ptr ptr ty') shape'
  where shape' = map scalarOp shape
        ty' = scalarTy ty

sizeOf :: [Operand] -> CompileM Operand
sizeOf shape = foldM mul (litInt 1) shape

mul :: Operand -> Operand -> CompileM Operand
mul x y = evalInstr "mul" longTy $ L.Mul False False x y []

scalarOp :: CompileVal -> Operand
scalarOp (ScalarVal op _) = op

addInstr :: Named Instruction -> CompileM ()
addInstr instr = modify $ setCurInstrs (instr:)

scalarTy :: BaseType -> L.Type
scalarTy ty = case ty of IntType  -> longTy
                         RealType -> realTy

baseTy :: L.Type -> BaseType
baseTy ty = case ty of
  L.IntegerType 64 -> IntType
  L.FloatingPointType L.DoubleFP -> RealType

compileBinop ::    L.Type -> (Operand -> Operand -> L.Instruction)
                -> [CompileVal]
                -> CompileM CompileVal
compileBinop ty makeInstr [ScalarVal x _, ScalarVal y _] =
  liftM (flip ScalarVal ty) $ evalInstr "" ty (makeInstr x y)

externalMono :: ExternFunSpec -> BaseType -> [CompileVal] -> CompileM CompileVal
externalMono f@(ExternFunSpec name retTy _ _) baseTy args = do
  ans <- evalInstr name' retTy $ externCall f (map scalarOp args)
  return $ ScalarVal ans (scalarTy baseTy)
  where name' = unpack (fromShort name)

compileBuiltin :: Builtin -> [CompileVal] -> CompileM CompileVal
compileBuiltin b = case b of
  Add      -> compileBinop longTy (\x y -> L.Add False False x y [])
  Mul      -> compileBinop longTy (\x y -> L.Mul False False x y [])
  Sub      -> compileBinop longTy (\x y -> L.Sub False False x y [])
  Hash     -> externalMono hashFun    IntType
  Rand     -> externalMono randFun    RealType
  Randint  -> externalMono randIntFun IntType
  _ -> error $ pprint b

randFun    = ExternFunSpec "randunif"      realTy [longTy] ["keypair"]
randIntFun = ExternFunSpec "randint"       longTy [longTy, longTy] ["keypair", "nmax"]
hashFun    = ExternFunSpec "threefry_2x32" longTy [longTy, longTy] ["keypair", "count"]
mallocFun  = ExternFunSpec "malloc_cod" charPtrTy [longTy] ["nbytes"]
memcpyFun  = ExternFunSpec "memcpy_cod" L.VoidType [charPtrTy, charPtrTy, longTy]
                                                   ["dest", "src", "nbytes"]

charPtrTy = L.ptr (L.IntegerType 8)
intPtrTy = L.ptr longTy
longTy = L.IntegerType 64
realTy = L.FloatingPointType L.DoubleFP

funTy :: L.Type -> [L.Type] -> L.Type
funTy retTy argTys = L.ptr $ L.FunctionType retTy argTys False

makeModule :: [NInstr] -> [BasicBlock] -> Module
makeModule decls (fstBlock:blocks) = mod
  where
    L.BasicBlock name instrs term = fstBlock
    fstBlock' = L.BasicBlock name (decls ++ instrs) term
    mod = L.defaultModule { L.moduleName = "test"
                          , L.moduleDefinitions =
                                L.GlobalDefinition fundef
                              : map externDecl
                                  [mallocFun, memcpyFun,
                                   hashFun, randFun, randIntFun]
                          }
    fundef = L.functionDefaults { L.name        = L.Name "thefun"
                                , L.parameters  = ([], False)
                                , L.returnType  = longTy
                                , L.basicBlocks = (fstBlock':blocks) }

externCall :: ExternFunSpec -> [L.Operand] -> L.Instruction
externCall (ExternFunSpec fname retTy argTys _) args =
  L.Call Nothing L.C [] fun args' [] []
  where fun = Right $ L.ConstantOperand $ C.GlobalReference
                         (funTy retTy argTys) (L.Name fname)
        args' = [(x ,[]) | x <- args]

externDecl :: ExternFunSpec -> L.Definition
externDecl (ExternFunSpec fname retTy argTys argNames) =
  L.GlobalDefinition $ L.functionDefaults {
    L.name        = L.Name fname
  , L.parameters  = ([L.Parameter t (L.Name s) []
                     | (t, s) <- zip argTys argNames], False)
  , L.returnType  = retTy
  , L.basicBlocks = []
  }

setScalarDecls update state = state { scalarDecls = update (scalarDecls state) }
setCurInstrs   update state = state { curInstrs   = update (curInstrs   state) }
setCurBlocks   update state = state { curBlocks   = update (curBlocks   state) }
setImpVarEnv   update state = state { impVarEnv   = update (impVarEnv   state) }
setBlockName   update state = state { blockName   = update (blockName   state) }

instance Functor JitVal where
  fmap = fmapDefault

instance Foldable JitVal where
  foldMap = foldMapDefault

instance Traversable JitVal where
  traverse f val = case val of
    ScalarVal x ty -> liftA (\x' -> ScalarVal x' ty) (f x)
    ArrayVal (Ptr ptr ty) shape ->
      liftA2 newVal (f ptr) (traverse f shape)
      where newVal ptr' shape' = ArrayVal (Ptr ptr' ty) shape'

instance Functor JitVals where
  fmap = fmapDefault

instance Foldable JitVals where
  foldMap = foldMapDefault

instance Traversable JitVals where
  traverse f (JitVals vals) = liftA JitVals $ traverse (traverse f) vals

newtype JitVals w = JitVals [JitVal w]