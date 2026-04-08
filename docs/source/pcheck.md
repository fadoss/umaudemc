(sec:pcheck)=
# Probabilistic model checking

Probabilistic model checking is available through the `pcheck` subcommand on Maude models extended with probabilities by some [probabilistic assignment method](#sec:probspec) specified with the `--assign` option.

```console
$ umaudemc pcheck ⟨filename⟩ ⟨initial state⟩ ⟨formula⟩ [ ⟨strategy⟩ ]
                  [ --assign ⟨method⟩ ] [ --reward ⟨term⟩ ]
```

The `formula` argument can be either `@steady` to calculate steady-state probabilities, `@transient(n)` for the transient probabilities at the `n`-th step, or a formula in LTL, CTL, or PCTL in the following syntax extending the standard `LTL` module:
%
```maude
*** CTL
op A_ : Formula -> Formula [ctor prec 53] .
op E_ : Formula -> Formula [ctor prec 53] .

*** bounded step operators
op _U__ : Formula Bound Formula -> Formula [ctor prec 63 …] .
op _R__ : Formula Bound Formula -> Formula [ctor prec 63 …] .

op <>__ : Bound Formula -> Formula [prec 53 …] .
op []__ : Bound Formula -> Formula [prec 53 …] .
op _W__ : Bound Formula Formula -> Formula [prec 63 …] .

*** bounded probability operator (PCTL)
op P__ : Bound Formula -> Formula [ctor prec 65 …] .
```

A formula $\mathbf{P}_I \, \varphi$ in PCTL holds when the probability that $\varphi$ is satisfied is in the interval $I \subseteq [0, 1]$. In the `P` operator of the Maude-based syntax, this interval is defined by a term of sort `Bound` built with

```maude
op <=_ : Float -> Bound [ctor] .       op <=_ : Nat -> Bound [ctor] .
op <_  : Float -> Bound [ctor] .       op <_  : Nat -> Bound [ctor] .
op >=_ : Float -> Bound [ctor] .       op >=_ : Nat -> Bound [ctor] .
op >_  : Float -> Bound [ctor] .       op >_  : Nat -> Bound [ctor] .

op [_,_] : Nat Nat -> Bound [ctor] .
op [_,_] : Float Float -> Bound [ctor] .
```

These bounds can also be attached to the temporal operators, although not all combinations are admitted by the backends. When checking an LTL or CTL formula, the probability that the property is satisfied will be calculated. Rewards for reachability formulas can also be computed as explained below.


In addition to the [general options](#sec:usage), some relevant modifiers are specific to this subcommand:

`--assign ⟨method⟩`
: Sets the probability assignment method to one of those explained in [](#sec:probspec). Instead of the literal description of the method, a filename prefixed by `@` may be entered to load it from file. If the selected method is `strategy`, the *strategy* argument must be filled. Otherwise, this argument is optional and the strategy would control the system in the standard way.

`--steps`
: For a reachability formula, calculates the expected number of steps instead of its probability.

`--reward ⟨term⟩`
: For a reachability formula, calculates the expected reward for the given term instead of its probability. The sort of this term should be a numerical one (`Int`, `Float`, `Integer`, `Real`, …) and it must contain at most one variable, which will be instantiated with every state to evaluate the reward. The `--steps` option can be seen as a shortcut for `--reward 1`.

`--raw-formula`
: The *formula* argument is directly passed to the backend, although it is scanned to find atomic propositions that should be evaluated in the model.

`--fraction`
: Results are printed as approximated fractions instead of decimal floating-point numbers.

## Examples

For example, we can calculate steady-state probabilities for the [philosophers example](#sec:models), which are only non-zero for the deadlock states.

```console
$ umaudemc pcheck philosophers.maude initial '@steady' --backend storm
 0.5                  < (ψ | 0 | o) (ψ | 1 | o) (ψ | 2 | o) >
 0.5                  < (o | 0 | ψ) (o | 1 | ψ) (o | 2 | ψ) >
```

We have forced Storm as backend because the probabilities obtained from PRISM are 0.49999880… due to approximation errors. Transient probabilities can also be obtained with

```console
$ umaudemc pcheck philosophers.maude initial '@transient(1)' --fraction \
    --assign 'uaction(left=2, right=3)'
 1/5                  < (o | 0 | o) ψ (o | 1 | o) ψ (o | 2 | ψ) >
 1/5                  < (o | 0 | ψ) (o | 1 | o) ψ (o | 2 | o) ψ >
 1/5                  < (o | 0 | o) ψ (o | 1 | ψ) (o | 2 | o) ψ >
 2/15                 < (ψ | 0 | o) ψ (o | 1 | o) ψ (o | 2 | o) >
 2/15                 < (o | 0 | o) ψ (o | 1 | o) (ψ | 2 | o) ψ >
 2/15                 < (o | 0 | o) (ψ | 1 | o) ψ (o | 2 | o) ψ >
```
%
The `--fraction` modifier has been used to see the probabilities as fractions, and `--assign` has selected the `uaction` method for assigning probabilities.

As examples of temporal properties, we can check the following:

```console
$ umaudemc pcheck philosophers.maude initial '<> eats(0)'
Result: 0.4999996389661516 (relative error 7.3569252716515036e-06)
$ umaudemc pcheck philosophers initial '<> <= 3 eats(0)'
Result: 0.24999999999999997
```

The unbounded one holds with probability 1 under the `parity` strategy, so we can also compute the expected number of steps until `eats(0)` is satisfied.

```console
$ umaudemc pcheck philosophers initial '<> eats(0)' parity --steps
Result: 4.799996844606567 (relative error 5.952947354729354e-06)
```

Assuming a function `eatCount` has been defined that counts how many philosophers are eating in a given state, we can also calculate the expected value of the reward:

```console
$ umaudemc pcheck philosophers.maude initial '<> eats(0)' parity \
    --reward '2 * eatCount(M)'
Result: 1.599999367157242 (relative error 7.855416516353985e-06)
```

(sec:pcheck-backends)=
## External backends and their installation

The `pcheck` command does not implement its own probabilistic model-checking algorithms, so an external tool needs to be installed. Two alternative backends are supported:

* [PRISM](https://www.prismmodelchecker.org/). There are installation instructions in its website. If not installed in the system path, its location should be provided using the `PRISM_PATH` environment variable.
* [Storm](https://www.stormchecker.org/). Its Python bindings, [StormPy](https://stormchecker.github.io/stormpy/), can be installed with `pip install stormpy`, and the backend can be directly used afterward. Alternatively, the `pcheck` command can also interact with the `storm` binary, whose location should be specified with the `STORM_PATH` environment variable if not available in the system path.

In general, StormPy should be more efficient than PRISM. The unified utility will choose the first available backend in the sequence specified with the following argument:

`–-backend ⟨list⟩` 
: Indicates a comma-separated list of probabilistic model-checking backends that will be used to check the given properties, among `prism` and `storm`. The default backend list follows this order.
