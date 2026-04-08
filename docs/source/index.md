# Unified Maude model-checking tool

The unified Maude model‑checking utility `umaudemc` is a uniform command‑line, graphical, and programming interface to different model checkers operating on Maude specifications. On both standard and strategy‑controlled Maude specifications, `umaudemc` can be used for

* checking LTL, CTL, CTL*, and μ‑calculus properties (see [](#sec:check)),
* applying probabilistic model‑checking methods (see [](#sec:pcheck)),
* estimating quantitative properties by statistical model checking (see [](#sec:scheck)), and
* exporting the rewrite graphs in different formats and more (see [](#sec:other-tools)).

This utility can be installed with
```console
$ pip install umaudemc
```
or downloaded from its [git repository](https://github.com/fadoss/umaudemc). It requires Python 3.9 or newer and the [`maude` Python library](https://github.com/fadoss/maude-bindings), which is automatically installed by the previous `pip` command. Some external backends are optional for [standard](#sec:check-backends) and required for [probabilistic](#sec:pcheck-backends) model checking, while installing [SciPy](https://scipy.org/) is advisable for statistical model checking. Follow the corresponding links for installation instructions.

```{seealso}
[Maude website](https://maude.cs.illinois.edu/) · [Maude 3.5.1 manual](https://maude.cs.illinois.edu/manual) · [Source code](https://github.com/fadoss/umaudemc)
```

(sec:usage)=
## Using the tool

The command‑line interface is organized in subcommands: [`check`](#sec:check), [`pcheck`](#sec:pcheck), [`scheck`](#sec:scheck), [`graph`](#sec:graph), and [`gui`](#sec:gui) Their syntax and options are listed by passing the `--help` flag. Almost every command starts with the following arguments:
```console
$ umaudemc ⟨subcommand⟩ ⟨Maude filename⟩ ⟨initial term⟩
```
Some more options are generally available for a more precise selection of the input Maude model:

`--module ⟨name⟩`, `-m ⟨name⟩`
: Selects the module specifying the system to be model checked. By default, as in
the Maude interpreter, the last module will be used unless a module is explicitly selected with a
select command in the file.

`--metamodule ⟨term⟩`, `-M ⟨term⟩`
: Selects the meta‑module described by its argument as the module where to model check. The term will be reduced in the module indicated by the module option or selected by default.

Moreover, the following arguments may be inserted between `umaudemc` and the subcommand:

`--verbose`, `-v`
: Select the verbose mode so that more information is printed by the tool.

`--no-advise`
: Suppress debug advisories from Maude, like the `-no-advise` option of the Maude interpreter.

`--version`
: Show the current version of the tool and the optional dependencies that have been detected to be available or missing.

Moreover, `umuademc` can be used as a Python library (see [](#sec:apiref)).

## Table of contents

```{toctree}
:maxdepth: 2

intro
check
pcheck
scheck
graph
biblio
apiref
```
