(sec:check)=
# Standard model checking

The `check` command checks LTL, CTL, CTL\*, and μ-calculus properties on standard and strategy-controlled Maude specifications. Several [alternative backends](#sec:check-backends) can be used (and may need to be installed) to check whether the given formula is satisfied, but the problem data is introduced and the verification result is shown in the same format for all of them. The basic syntax of the command is:

```console
$ umaudemc ⟨filename⟩ ⟨initial state⟩ ⟨formula⟩ [ ⟨strategy⟩ ]
```

The optional strategy argument can be any well-defined strategy expression. Formulae are written in the language of the `LTL` module extended with path quantifiers for CTL\* and μ-calculus operators as explained in [](#sec:check-formulas). 

In addition to these positional arguments, other optional parameters can be set up that are listed when invoking the program with the `--help` flag. By default, the module that would be selected when loading the given file into the Maude interpreter will be the selected module for model checking (i.e. the last module to be parsed or selected with the `select` command), but this can be changed with the following options:

`--module ⟨name⟩`, `-m ⟨name⟩`
: Select a module by its name to be used for model checking.

`–-metamodule ⟨term⟩`, `-M ⟨term⟩` 
: Select the module whose metarepresenation is provided as an argument for model checking. The given term will be evaluated in the module specified with `--module` or selected by default.

Other options are used to control the format of counterexamples. The `-–slabel` option is specially useful to simplify counterexamples involving complex terms.

`-c`
: Prefers backends that provide counterexamples to those that do not provide them.

`–-slabel ⟨format⟩` 
: Sets the format of state labels to a string that may contain special variables: `%t` for the current term, `%s` for the immediate strategy continuation, and `%i` for the internal state index. Moreover, arbitrary Maude terms containing `%t` can be written between curly brackets to be evaluated and replaced by their results. For example, an atomic proposition can be checked with `{%t |= aprop}`).

`–-elabel ⟨format⟩`
: Sets the format of edge labels to a string that may contain special variables: `%s` for the transition statement (rule, strategy declaration...), `%l` for its label, `%n` for its line number, and `%o` for `opaque` if the transition is caused by an opaque strategy.

`–-format ⟨option⟩` 

: Determines how counterexamples are formatted. By default, they are printed as colored text (`text`), but they can also be written in JSON (`json`), GraphViz's DOT (`dot`) and HTML (`html`).

Special variables in `slabel` and `elabel` can be truncated to a specified length $n$ by writing {codemath}`.$n$` between the `%` sign and the letter.

(sec:check-formulas)=
## Syntax of formulas

Formulae are written in the language of the `LTL` module (see [](#sec:modelprep)) extended with path quantifiers for CTL\* and μ-calculus operators.

``` maude
sorts @MCVariable@ @Action@ @ActionSpec@ @ActionList@ .
subsort @MCVariable@ < Formula .

*** CTL and CTL*
op A_ : Formula -> Formula [prec 53 …] .
op E_ : Formula -> Formula [prec 53 …] .

*** mu-calculus
op <.>_  : Formula -> Formula [prec 53 …] .
op [.]_  : Formula -> Formula [prec 53 …] .
op <_>_  : @ActionSpec@ Formula -> Formula [prec 53 …] .
op [_]_  : @ActionSpec@ Formula -> Formula [prec 53 …] .
op mu_._ : @MCVariable@ Formula -> Formula [prec 64 …] .
op nu_._ : @MCVariable@ Formula -> Formula [prec 64 …] .

subsorts @Action@ < @ActionList@ < @ActionSpec@ .
op opaque : @Action@ -> @ActionList@ [ctor] .
op ~_ : @ActionList@ -> @ActionSpec@ [ctor prec 50] .
op __ : @ActionList@ @ActionList@ -> @ActionList@ [ctor assoc] .
```

`A` and `E` are the universal and existential path quantifiers on CTL and CTL\* formulae. The language of μ-calculus includes the existential modalities `<.>` and `<_>`, the universal modalities `[.]` and `[_]`, and the fixed-point operators `mu` and `nu`, as well as all other propositional-logic operators defined in `LTL`. Modalities taking an `@ActionSpec@` as argument must include a space-separated list of rule labels, which can be prefixed by the `~` symbol to indicate its complement. For opaque strategies (see [](#sec:opaque)), their names should be specified with the `opaque` constructor. The steps described by the modality are those labeled with one of these symbols, as usual.

$$\langle\, l_1 \cdots\, l_n \,\rangle\; \varphi \coloneq \bigvee_{i=1}^n \,\langle\, l_k \,\rangle\; \varphi \qquad\qquad [\, l_1 \cdots\, l_n \,]\; \varphi \coloneq \bigwedge_{i=1}^n \,[\, l_k \,]\; \varphi$$

The dot version of the same operators considers all possible transitions, as if the whole list of rule labels was written. Fixed-point operators are followed by a variable name that should be used in the nested subformula. Any identifier can be a variable as long as it does not conflict with other syntactical elements of the formula. The program will parse the given formula in this grammar,[^1] and deduce the logic in which they are expressed and all required information to call the appropriate model checker. Formulae mixing μ-calculus and CTL\* operators are not valid.

(sec:check-strat)=
## Strategy-aware model checking

If the `strategy` argument is given in the command line, properties are checked for the corresponding strategy-controlled model. When branching-time properties are used, the rewriting graph generated for LTL model checking must be applied some adaptations {cite:p}`smcJournal-btime`. This utility will automatically choose the appropriate ones according to the input problem data, but the user may still overwrite the default settings. These options are meaningless without strategies, and so will be ignored in that case.

`–-merge-states ⟨option⟩` 
: Merges successor states with a common term but different strategy continuations, if the option is `state` or `edge`. Moreover, with `edge` only successors by a common transition label are merged. Merging can be disabled completely with `no` and can be instructed to set the appropriate configuration with `default` (`edge` for μ-calculus with edge labels, `no` for LTL, and `state` for the rest).

`–-purge-fails ⟨option⟩` 
: Enables (`on`) or disables (`off`) the elimination of failed states. The `default` option is `off` for LTL and on-the-fly model checking algorithms, and `on` otherwise.

`–-opaque ⟨list⟩`
: Comma-separated list with the names of the strategies to be considered [*opaque*](#sec:opaque) (i.e. their complete execution is an atomic step of the model).

`–-kleene-iteration`, `-k`
: Make the semantics of the iteration coincide with the Kleene star (see [](#sec:kleene)).

`–-show-strat`
: Shows the next strategy to be executed from each state in the counterexample.

## Examples

On the [dining philosophers example](#sec:models), we can check the CTL formula $\mathbf{A} \,\square\, \mathbf{E} \,\diamond\,  \mathtt{eats(0)}$ with the following command:

```{erbsland-ansi}
$ umaudemc check philosophers.maude initial 'A [] E <> eats(0)'
The property [1m[31mis not[0m satisfied in the initial state
(27 system states, 2734 rewrites)
```

The property is not satisfied, but we are not shown a counterexample. Among the supported backends, only NuSMV provides counterexamples for CTL properties, but it is not the first one in the default ordered list of model-checking backends. We can use the `-c` flag to prefer a backend that provides counterexamples, if available.

```{erbsland-ansi}
$ umaudemc check philosophers initial 'A [] E <> eats(0)' -c
The property [1m[31mis not[0m satisfied in the initial state
(27 system states, 98 rewrites)
[31m|[0m < (o | 0 | o) ψ (o | 1 | o) ψ (o | 2 | o) ψ >
[31m∨[0m  [3;36mrl < (o | Id | X) L ψ > => < (ψ | Id | X) L > [label left] .[0m
[31m|[0m < (ψ | 0 | o) ψ (o | 1 | o) ψ (o | 2 | o) >
[31m∨[0m  [3;36mrl ψ (o | Id | X) => ψ | Id | X [label left] .[0m
[31m|[0m < (ψ | 0 | o) ψ (o | 1 | o) (ψ | 2 | o) >
[31m∨[0m  [3;36mrl ψ (o | Id | X) => ψ | Id | X [label left] .[0m
[1m[36mO[0m < (ψ | 0 | o) (ψ | 1 | o) (ψ | 2 | o) >
```

For branching-time properties, a counterexample (or an example for an existential property) cannot be a full execution, but the execution prefix *matched* by the first path quantifier can be obtained. For instance, the counterexample above shows a path to a state where no path satisfying `<> eats(0)` leaves. However, this property is satisfied if the system is controlled by the `parity` strategy in [](#sec:modelprep).

```{erbsland-ansi}
$ umaudemc check philosophers initial 'A [] E <> eats(0)' parity
The property [1m[32mis[0m satisfied in the initial state
(12 system states, 328 rewrites)
```

To alternatively see whether the `release` rule is eventually executed, we can check the following μ-calculus property:

```{erbsland-ansi}
$ umaudemc check philosophers initial \
    'mu X . (< release > True \/ (<.> True /\ [.] X))' parity
The property [1m[32mis[0m satisfied in the initial state
(18 system states, 104 rewrites, 129 game states)
```

Moreover, the tool can be used to verify linear-time properties as well. In this case, we add the `elabel` and `slabel` options to simplify the printed counterexample.

```{erbsland-ansi}
$ umaudemc check philosophers.maude initial \
   '[] (<> eats(0) /\ <> eats(1) /\ <> eats(2))' parity \
   --elabel %l --slabel 'eats(1) = {%t |= eats(1)}'
The property [31;1mis not[0m satisfied in the initial state
(5 system states, 136 rewrites, 4 Büchi states)
[31m| |[0m eats(1) = false
[31m| ∨[0m  [3;36mleft[0m
[31m| |[0m eats(1) = false
[31m| ∨[0m  [3;36mright[0m
[31m| |[0m eats(1) = false
[31m| ∨[0m  [3;36mrelease[0m
[31m< ∨[0m
```

The strategy argument admits arbitrary strategy expressions to control the system.

```{erbsland-ansi}
$ umaudemc check philosophers.maude initial '[] allEat(3)' 'turns(0, 3)'
The property [1;32mis[0m satisfied in the initial state
(10 system states, 128 rewrites, 4 Büchi states)
```

[^1]: The sorts `@Action@` and `@MCVariable@` of the grammar are dynamically populated with the labels of the selected module and the candidate variables in the formula.

(sec:check-backends)=
## External model checkers and their installation

The `umaudemc` utility does model checking by alternatively calling the builtin Maude model checker, some algorithms implemented in the tool itself, and some external model checkers. LTL, CTL and μ-calculus properties can be checked out of the box using the first two options. However, for using other model checkers like LTSmin, NuSMV, Spot, or Spin their corresponding programs have to be installed or downloaded in locations where the utility is able to find them. Some backends may be more efficient that others (in general, the builtin Maude model checker and LTSmin should be the best choices), some easier to install, and not all of them support all temporal logics, as shown in the following table.

| Tool	          | Backend   | LTL        | CTL	| CTL*	 | μ-calculus  |
| --------------- | --------- | :--------: | :----: | :----: | :---------: |
| Maude			  | `maude`   | on-the-fly |        |	     |             |
| LTSmin		  | `ltsmin`  | on-the-fly | ✅     | ✅	 | ✅          |
| pyModelChecking |	`pymc`    | tableau    | ✅     | ✅	 |             |
| NuSMV           |	`nusmv`   | tableau    | ✅     | 	     |             |
| Spot            |	`spot`    | automata   |        | 	     |             |
| Spin            |	`spin`    | automata   |        | 	     |             |
| Own tools       | `builtin` |            | ✅     | 	     | ✅          |

* The built-in LTL model checker included in Maude {cite:p}`maudemc` and its extension for strategy-controlled systems {cite:p}`ause` are already available in the `maude` library. No action is required to install them.
* [LTSmin](https://github.com/utwente-fmt/ltsmin) is a language-independent model checker, for which a Maude language extension has been written {cite:p}`smcJournal-btime`. Download the model checker [here](https://github.com/utwente-fmt/ltsmin/releases/), extract the package, and set the environment variable `LTSMIN_PATH` to its binary directory. The environment variable `MAUDEMC_PATH` should also be set to the full path of the Maude-LTSmin plugin that can be downloaded from [maude.ucm.es/strategies](https://maude.ucm.es/strategies/#downloads). However, for model checking μ-calculus properties containing both atomic propositions and rule labels, a modified version of LTSmin is required as well as the `pbespgsolve` tool from mCRL2 {cite:p}`mCRL2`. A ready-to-use package[^proposed] can be downloaded from [here](https://maude.ucm.es/strategies).
* [NuSMV](https://nusmv.fbk.eu) is well-known BDD-based model checker for LTL and CTL. After downloading it from [here](https://nusmv.fbk.eu/downloads.html), extract the package and set the environment variable `NUSMV_PATH` to the path where the `NuSMV` binary resides.
* [`pyModelChecking`](https://github.com/albertocasagrande/pyModelChecking) is a simple Python model-checking library. It can be installed with `pip install pyModelChecking`.
* [Spot](https://spot.lrde.epita.fr/) is a library and tool for LTL and ω-automata manipulation. There are [installation instructions](https://spot.lre.epita.fr/install.html) in its website. Once the Python package is installed, it will be available for `umaudemc`.
* [Spin](https://spinroot.com/) is a widely-used LTL model checker. The path containing the `spin` binary should be specified with the `SPIN_PATH` environment variable if not already in the system path.
* The `umaudemc` tool includes a built-in μ-calculus implementation based on the procedure described [here](https://doi.org/10.1007/978-3-319-10575-8_26) and Zielonka's algorithm. CTL is also supported by a syntactic translation.

Given a temporal property, `umaudemc` will choose the first supported backend available to model check it. They are chosen in the order they appear in the table, which can nevertheless be modified with the following option:

`–-backends ⟨list⟩` 
: Indicates a comma-separated list of model-checking backends that will be used to check the given properties, among `maude`, `ltsmin`, `pymc`, `nusmv`, `spin`, `spot`, and `builtin`. The default backend list follows this order.

All arguments passed to `umaudemc` after a pair `--` of dashes will be passed directly to the backends, in case they are external programs (LTSmin, NuSMV, and Spin).

[^proposed]: The modification has been proposed to be included in the upstream LTSmin. However, LTSmin does not seem to be in active development since 2024.

