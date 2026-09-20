# rhymes

Turn a nursery rhyme you wrote into a sung audio track, for free, without owning a GPU.

Write the words. Pick a style by name. Get an mp3.

```
$ rhymes render twinkle.txt
twinkle.mp3
```

It wraps [HeartMuLa](https://github.com/HeartMuLa/heartlib) (Apache 2.0), which does the
actual singing. This is the thin part: nursery-rhyme defaults, a style table you don't
have to learn a tag vocabulary for, checkpoint bootstrap, and errors that arrive in a
second instead of after a model load.

## Run it on a free GPU

Generation needs CUDA, so the supported path is
[`notebooks/rhymes_colab.ipynb`](notebooks/rhymes_colab.ipynb) on a free Colab T4. Open
it, set the runtime to a T4 GPU, and Run All. No cell needs editing.

The first render downloads 22.4 GB of checkpoints (15.75 GB of 3B weights, 6.64 GB of
codec). A second render in the same runtime reuses them.

**Apple silicon is not supported.** `--device` exists as an escape hatch and warns you it
is unsupported. Nothing in heartlib needs CUDA kernels, so it is not obviously impossible
— but it is untested here, and two things make it hard: HeartMuLa is quoted at roughly
realtime on a datacenter GPU, so expect many multiples of that; and `lazy_load` is forced
off on non-CUDA devices (heartlib's unload path calls `torch.cuda` unconditionally), which
means the 3B weights in bf16 plus the fp32 codec — about 14.5 GB — must all stay resident.


## Writing lyrics

Section headers, plain lines underneath:

```
[Verse]
Twinkle twinkle little star
How I wonder what you are

[Chorus]
Up above the world so high
Like a diamond in the sky
```

Markdown headers work too, since that is what you get when you draft lyrics in a chat or
an editor — `**Verse**`, `__Verse__` and `## Verse` are all read as headers. A line is
only treated as a header if the whole line matches, so `I **really** like bananas` stays
a lyric.

The model knows six section names: **Intro, Verse, Prechorus, Chorus, Bridge, Outro**.
Anything else (`[Verse 1]`, `[Dance Break]`) still renders, but reaches the model as raw
text and may not land — you get a one-line note on stderr when that happens.

Emoji are not singable. They are passed through untouched rather than stripped, because
the tool never edits your words, but you probably want them out of the lyrics.

Anything malformed is rejected before the model loads, with the offending line number.

## Usage

```
rhymes styles                                 # list presets and what they expand to
rhymes render LYRICS.txt                      # writes LYRICS.mp3 beside the input
rhymes render LYRICS.txt -o out.mp3 --style lullaby
rhymes render LYRICS.txt --tags "piano,gentle,children"
rhymes render LYRICS.txt --seconds 45 --temperature 0.8
```

`--style` and `--tags` are mutually exclusive. `--seconds` accepts 5 to 300, but long
renders will exhaust a free T4's 16 GB.

Exit codes: `0` success, `1` a validation or runtime failure with a one-line reason,
`2` a usage error.

## Install

Python 3.10 or newer. On 3.13+ the install compiles `numpy==2.0.2` (a heartlib pin)
from source, because that release has no 3.13 wheel — slower, but it works, which is what
Colab does today.

```
pip install git+https://github.com/somashekar4138/rhymes_gen
pip install "rhymes[gpu] @ git+https://github.com/somashekar4138/rhymes_gen"   # adds heartlib
```

Set `RHYMES_CACHE_DIR` to put checkpoints somewhere other than `~/.cache/rhymes/ckpt`.

For development, including the tools the gate runs:

```
uv venv --python 3.12
uv pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy . && pytest -q
```

## Your lyrics stay put

The only network call this package can make is downloading model checkpoints. No
telemetry, no analytics, no remote generation API. `tests/test_no_egress.py` parses the
package's own source and fails if that stops being true.

## Licence

Apache 2.0. See [NOTICE](NOTICE) for heartlib's attribution.
