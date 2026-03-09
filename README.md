# betterdf

Ergonomic DataFrame methods for pandas. Monkey-patches `pd.DataFrame` with strict, expressive helpers for column selection, row filtering, type checking, and row-wise transforms.

## Installation

```bash
pip install betterdf
```

## Quick start

```python
import betterdf
import pandas as pd

betterdf.patch()  # adds methods to pd.DataFrame

df = pd.DataFrame({
    "name": ["alice", "bob", "carol"],
    "age": [30, 25, 35],
    "score": [88.5, 92.0, 76.3],
})

# Select / drop columns (strict — raises on typos)
df.select("name", "age")
df.deselect("score")

# Filter rows with a lambda
df.keep(lambda r: r.age > 25)

# Add or overwrite columns
df.mutate(grade=lambda r: "A" if r.score >= 90 else "B")

# Row-wise transform with progress bar
df.vapply(lambda r: pd.Series({"name": r.name.upper(), "age": r.age}))
```

## API

### `df.select(*columns) -> DataFrame`

Keep only the named columns. Raises `KeyError` if any column is missing.

```python
df.select("name", "age")
```

### `df.deselect(*columns) -> DataFrame`

Drop the named columns. Raises `KeyError` if any column is missing.

```python
df.deselect("score")
```

### `df.keep(fn) -> DataFrame`

Keep rows where `fn(row)` is truthy. The row is a `StrictSeries` — accessing a nonexistent column raises immediately.

```python
df.keep(lambda r: r.age >= 30)
```

### `df.mutate(**columns) -> DataFrame`

Add or overwrite columns using row-level functions. Each function receives a `StrictSeries` row and returns a scalar.

```python
df.mutate(
    bmi=lambda r: r.weight / r.height ** 2,
    label=lambda r: "tall" if r.height > 180 else "short",
)
```

### `df.vapply(fn) -> DataFrame`

Serial row-wise apply with a tqdm progress bar. `fn` receives a `StrictSeries` and returns a `pd.Series`. Return `betterdf.DROP` to drop a row.

```python
import betterdf

def transform(r):
    if r.score < 50:
        return betterdf.DROP
    return pd.Series({"name": r.name, "passed": True})

df.vapply(transform)
```

### `df.papply(fn, n_workers=None) -> DataFrame`

Parallel version of `vapply` using threads. Row order is preserved. Defaults to `os.cpu_count()` workers.

```python
df.papply(transform, n_workers=4)
```

### `df.assert_types(**expected) -> DataFrame`

Assert column dtypes loosely (e.g. `float` matches float32/float64). Returns self for chaining. Raises `TypeError` on mismatch.

```python
df.assert_types(price=float, name=str, count=int)
```

### `df.cast(**types) -> DataFrame`

Cast columns to specified types. Thin wrapper around `astype` with column validation.

```python
df.cast(price=float, count=int)
```

### `df.relabel(**mappings) -> DataFrame`

Recode values in one or more columns. Unmapped values are left unchanged.

```python
df.relabel(
    region={"N": "North", "S": "South", "E": "East", "W": "West"},
    status={"A": "Active", "I": "Inactive"},
)
```

### `df.collapse(*group_cols, name="records") -> DataFrame`

Group by columns and collect remaining columns as a list of dicts per group.

```python
df.collapse("region", "year")
# Each row has a "records" column containing a list of dicts
```

### `df.peek(name=None) -> DataFrame`

Print shape and first 2 rows, then return self for chaining. Useful for debugging pipelines.

```python
result = df.keep(lambda r: r.age > 25).peek("filtered").select("name", "age")
```

## StrictSeries

All row-level functions (`keep`, `mutate`, `vapply`, `papply`) receive a `StrictSeries` instead of a raw `pd.Series`. It raises `AttributeError` or `KeyError` immediately if you access a nonexistent column — no silent `NaN` values.

```python
df.keep(lambda r: r.naem > 5)  # AttributeError: Column 'naem' not found
```

## Patching and unpatching

```python
betterdf.patch()    # add all methods to pd.DataFrame
betterdf.unpatch()  # remove them
```
