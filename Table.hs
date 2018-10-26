module Table (fromScalar, toScalar, diag, mapD, mapD2, reduceD, iota,
              printTable, insert, Table (..)) where

import Prelude hiding (map, lookup)
import Data.List (intersperse, transpose, sortOn)
import Util
import qualified Prelude as P
import qualified Data.Map.Strict as M


data Table a b = Table Int [([Maybe a], b)]
data MMap k v = MMap (M.Map k v) | Always v
type T = Table

fromScalar :: Ord a => b -> T a b
fromScalar x = Table 0 [([], x)]

toScalar :: Ord a => T a b -> b
toScalar (Table n [([], x)]) | n == 0 = x

insert :: Int -> T k a -> T k a
insert pos (Table n rows) =
  let update = insertIdx pos Nothing
  in Table (n+1) [(update ks, v) | (ks, v) <- rows]

mapD ::  Ord k => Int -> (T k a -> T k b) -> T k a -> T k b
mapD d = composeN d map

map ::  Ord k => (T k a -> T k b) -> T k a -> T k b
map f t = fromMMap $ map' f (toMMap "4" t)

reduceD :: Ord k => Int -> (T k a -> T k a -> T k a) -> T k a -> T k a -> T k a
reduceD d f z xs = mapD2 d (reduce f) z xs

reduce :: Ord k => (T k a -> T k a -> T k a) -> T k a -> T k a -> T k a
reduce f z xs = reduce' f z (toMMap "3" xs)

mapD2 :: Ord k => Int -> (T k a -> T k b -> T k c) -> T k a -> T k b -> T k c
mapD2 d = composeN d map2

map2 :: Ord k => (T k a -> T k b -> T k c) -> T k a -> T k b -> T k c
map2 f x y = fromMMap $ map2' f (toMMap "1" x) (toMMap "2" y)

toMMap :: Ord k => String -> T k v -> MMap k (T k v)
toMMap s (Table 0 [([],_)]) = error $ "Can't express scalar table as map" ++ s
toMMap s (Table n rows) | n > 0 =
    let peelIdx (k:ks, v) = (k, (ks, v))
    in case group $ P.map peelIdx rows of
        [(Nothing, rows')] -> Always $ Table (n-1) rows'
        groupedRows -> let rows' = [(unJust k, Table (n-1) v) | (k, v) <- groupedRows]
                       in MMap (M.fromList rows')

fromMMap :: Ord k => MMap k (T k v) -> T k v
fromMMap (Always t) = insert 0 t
fromMMap (MMap m)   = let rows = [(Just k : ks, v) | (k, Table _ rows) <- M.toList m
                                                   , (ks, v) <- rows]
                          (_, Table n _):_ = M.toList m
                      in Table (n+1) $ (sortOn fst) rows

iota :: T Int Int -> T Int Int
iota (Table n [([], v)]) = Table (n+1) [([Just i], i) | i <- [0..(v-1)]]

diag :: Ord k => Int -> Int -> Table k a -> Table k a
diag 0 0 t = t
diag 0 j (Table n rows) | n > 0 = Table (n-1) . sortOn fst . mapMaybe mergeRow . mapFst (mvIdx j 1) $ rows
diag i j t = mapD i (diag 0 (j-i)) t

mergeRow :: Ord k => ([Maybe k], v) -> Maybe ([Maybe k], v)
mergeRow ((Nothing : Nothing : ks), v)             = Just (Nothing :ks, v)
mergeRow ((Nothing : Just k  : ks), v)             = Just ((Just k):ks, v)
mergeRow ((Just k' : Just k  : ks), v) | k == k'   = Just ((Just k):ks, v)
                                       | otherwise = Nothing

-- -- ----- operations on maps -----

map' :: (a -> b) -> MMap k a -> MMap k b
map' f (Always v) = Always $ f v
map' f (MMap m) = MMap $ M.map f m

map2' :: Ord k => (a -> b -> c) -> MMap k a -> MMap k b -> MMap k c
map2' f (Always x) (Always y) = Always $ f x y
map2' f (Always x) (MMap  my) = MMap $ M.map (f x) my
map2' f (MMap  mx) (Always y) = MMap $ M.map (flip f $ y) mx
map2' f (MMap  mx) (MMap  my) = MMap $ M.intersectionWith f mx my


reduce' :: (v -> v -> v) -> v -> MMap k v -> v
reduce' f z (Always x) = error "Can't reduce infinite map"
reduce' f z (MMap mx ) = M.foldr f z mx

-- -- ----- printing -----

printTable :: (Show a, Show b) => Table a b -> String
printTable (Table n rows) = concat . P.map formatRow . rowsToStrings $ rows

showMaybe :: (Show a) => Maybe a -> String
showMaybe Nothing = "*"
showMaybe (Just x) = show x

rowsToStrings :: (Show a, Show b) => [([Maybe a], b)] -> [[String]]
rowsToStrings rows =
  let stringRows = [[showMaybe k | k <- ks] ++ [show v] | (ks,v) <- rows]
      evalMaxLen = foldr (\s w -> max (length s) w) 0
      ws = P.map evalMaxLen . transpose $ stringRows
      padRow xs = [padLeft w ' ' x | (w, x) <- zip ws xs]
  in P.map padRow stringRows

formatRow :: [String] -> String
formatRow rows = " " ++ concat (intersperse " | " rows) ++ "\n"