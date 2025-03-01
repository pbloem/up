## frauenfrage.txt

See source here: https://www.gutenberg.org/ebooks/14075 Chosen to be different from WP (a different language and older) 
but still containing some structured information (tables and numbers) in addition to natural language.

## code.js

The minified version of D3js Source here: https://d3js.org/getting-started#d3-in-vanilla-html Code 
license: https://github.com/d3/d3/blob/main/LICENSE#L4C1-L13C15

## linux.gz

The first 200M bytes of the concatenated linux source code. Collected as follows:
```
find linux-master -type f -exec cat '{}' \; | cat > ./linux.txt
head -c 200000000 linux.txt > linux-head.txt && gzip linux-head.txt
```
Where linux master is the unzipped archive downloaded from Github: https://github.com/torvalds/linux

The current version was downloaded on 27 Feb 2025 14:43 and was at commit dd83757.

During loading, we collapse all whitespace into a single space with 
```python
with gzip.open(here('./up/data/linux.txt.gz'), 'r') as file:
    lin = file.read()
lin = lin.decode('utf-8', errors='replace')
lin = re.sub('\s+', ' ', lin)
```