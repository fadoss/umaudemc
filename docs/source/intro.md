(sec:intro)=
# Introduction

Maude {cite:p}`maude` let users specify concurrent and complex systems by means of terms, equations and rules, the language of rewriting logic {cite:p}`rewritingLogic`. These models can be directly executed or simulated by the interpreter, or analyzed by various tools. One of them is a built-in model checker {cite:p}`maudemc` for linear temporal logic (LTL) properties, which was first extended to support strategy-controlled models {cite:p}`fscd,ause`. Since rewriting is just the application of rules one after the other, in any order and anywhere, Maude incorporated a strategy language {cite:p}`theslang` to gain some global control on the process. It is designed as a new specification layer on top of equations and rules, so that the same model can exhibit different behaviors when controlled by alternative strategies. Both standard and strategy-controlled systems can be qualitatively checked using the `umaudemc` tool against LTL, CTL, CTL*, and μ-calculus properties through builtin and external tools {cite:p}`smcJournal-btime`, all integrated into the unified Maude model checking tool (`umaudemc`).

In addition to qualitative model checking, the `umaudemc` tool supports quantitative model checking with probabilistic and statistical methods {cite:p}`qmaude`. The same models where qualitative properties have been checked can be [extended with quantitative information](#sec:probspec) to check quantitative properties with the [`pcheck`](#sec:pcheck) and [`scheck`](#sec:scheck) commands.

This introduction starts with a short discussion on how models to be model checked are specified in Maude. The procedure does not differ much from the way it is done in the standard model checker described in Chapter 12 of the Maude manual {cite:p}`maude`. [Strategy-controlled](#sec:smodels) and [probabilistic models](#sec:probspec) are then explained.

```{seealso}
This introduction is based on the document [Maude strategy‑aware, external, and quantitative model checkers user manual](https://maude.ucm.es/strategies/modelchecker-manual.pdf), which can be downloaded in PDF from the previous link.
```

(sec:models)=
## Standard Maude models

Models in model checking are formalized as annotated state and transition systems known as Kripke structures. Rewriting systems can be naturally viewed as Kripke structures by identifying terms with states and adding a transition from one state to another if the first can be rewritten to the second by a rule. Like this, the executions of the model are sequences of rule applications. However, since temporal properties are usually only defined on infinite executions, the one-step rewrite relation should be completed by adding self-loops to all *deadlock* states, where no transition leaves. This is how the standard Maude LTL model checker works {cite:p}`maude`.

Let us introduce an example for explaining, in the following sections, how Maude specifications are prepared for model checking. The classical problem of the *dining philosophers* is specified in the following modules: a group of $n$ philosophers is gathered around a table to have dinner, for what they have to take the forks at both their sides. However, the table is round and there are only $n$ forks, so they cannot eat all at the same time.

```maude
fmod PHILOSOPHERS-TABLE is          *** functional module
    protecting NAT .

    sorts Obj Phil Being List Table .
    subsorts Obj Phil < Being < List .

    op (_|_|_) : Obj Nat Obj -> Phil [ctor] .
    ops o ψ : -> Obj [ctor] .
    op empty : -> List [ctor] .
    op __ : List List -> List [ctor assoc id: empty] .
    op <_> : List -> Table [ctor] .

    var L : List .
    ceq < ψ L > = < L ψ > if L =/= empty .
    op initial : -> Table .
    eq initial = < (o | 0 | o) ψ (o | 1 | o) ψ (o | 2 | o) ψ > .
endfm

mod PHILOSOPHERS-DINNER is          *** system module
    protecting PHILOSOPHERS-TABLE .
    var Id : Nat .
    var X : Obj .
    var L : List .
    rl [left] : ψ (o | Id | X) => (ψ | Id | X) .
    rl [right] : (X | Id | o) ψ => (X | Id | ψ) .
    rl [left] : < (o | Id | X) L ψ > => < (ψ | Id | X) L > .
    rl [release] : (ψ | Id | ψ) => ψ (o | Id | o) ψ .
endm
```
The philosophers can try three different moves: taking their `left` fork, their `right` fork, or `release` both of them at once. Doing it at their sole discretion may lead to some unwanted situations, like the starvation of some of them, or worse, of all of them. Strategies can prevent some of these problems by imposing additional restrictions, as explained [soon](#sec:smodels).

(sec:modelprep)=
### Model preparation

The following ingredients should be supplied to prepare any Maude specification for model checking:
supplied:

* A model specified by a system module `M` with a designated sort of states.
* A set of atomic propositions and a satisfaction relation $\vDash$ that specifies which are satisfied in each state.
* A temporal formula built on top of the previous atomic propositions.
* An initial term.

For instance, in the dining philosophers example, the rewriting model is given by the system module `PHILOSOPHERS-DINNER`, whose sort `Table` should be designated as the state sort. The initial term can be the `initial` symbol, where all the forks are on the table.

```{tikz} Structure of the model checker modules.
:align: center

[node distance=.7cm and .8cm, font=\ttfamily]

\node[draw] (SAT) {SATISFACTION};
\node[draw, left=of SAT] (QID) {QID};
\node[draw, left=of QID] (LTL) {LTL};

\node[draw, above=of QID] (BOOL) {BOOL};

\node[draw, below=of QID] (CH) {MODEL-CHECKER};

\node[draw, right=4em of SAT] (MP) {M-PREDS};
\node[draw, above=of MP](M) {M};
\node[draw, below=of MP] (MC) {M-CHECK};

\draw[->] (SAT) -- (MP);
\draw[double, ->] (M) -- (MP);
\draw[double, ->] (M) -- (MP);
\draw[double, ->] (MP) -- (MC);
\draw[->] (LTL) -- (CH);
\draw[->] (SAT) -- (CH);
\draw[double, ->] (QID) -- (CH);
\draw[->] (CH) -- (MC);
\draw[double, ->] (BOOL) -- (LTL);
\draw[double, ->] (BOOL) -- (QID);
\draw[double, ->] (BOOL) -- (SAT);
```

The procedure involves some predefined and user-defined modules respectively depicted in the left and right sides of the diagram. For the predefined modules, the `model-checker.maude` file in the Maude standard distribution should be loaded. The state sort is selected together with the declaration of the atomic propositions and their satisfiability. This is usually done in a new system module, say `M-PREDS`, that includes `M` and the predefined module `SATISFACTION`, where the `State` and `Prop` sorts, and the satisfaction relation symbol `_|=_` are declared.

```maude
fmod SATISFACTION is
    protecting BOOL .
    sorts State Prop .
    op _|=_ : State Prop -> Bool [frozen] .
endfm
```

The sort of states is selected by making it a subsort of `State`. `M-PREDS` must be a protected extension of `M` to ensure that the model is not altered in any way (although this is not checked by Maude). Finally, a system module `M-CHECK` merges the specification of the model and properties in `M-PREDS` with the predefined `MODEL-CHECKER` module. This module gives access to the `modelCheck` symbol and transitively imports the `LTL` module where the syntax for this temporal logic is defined.

``` maude
*** primitive LTL operators
ops True False : -> Formula [...] .
op ~_ : Formula -> Formula [prec 53 ...] .
op _/\_ : Formula Formula -> Formula [comm prec 55 ...] .
op _\/_ : Formula Formula -> Formula [comm prec 59 ...] .
op O_ : Formula -> Formula [prec 53 ...] .
op _U_ : Formula Formula -> Formula [prec 63 ...] .
op _R_ : Formula Formula -> Formula [prec 63 ...] .

*** defined LTL operators
op _->_ : Formula Formula -> Formula [prec 65 ...] .
op _<->_ : Formula Formula -> Formula [prec 65 ...] .
op <>_ : Formula -> Formula [prec 53 ...] .
op []_ : Formula -> Formula [prec 53 ...] .
op _W_ : Formula Formula -> Formula [prec 63 ...] .
op _|->_ : Formula Formula -> Formula [prec 63 ...] . *** leads-to
op _=>_ : Formula Formula -> Formula [prec 65 ...] .
op _<=>_ : Formula Formula -> Formula [prec 65 ...] .
```

An optional `LTL-SIMPLIFIER` module can be included to simplify the LTL formula. In practice, there is no need to follow this module structure and `M`, `M-PREDS`, and `M-CHECK` can be written as a single module, but this is the usual convention.

Coming back to the dining philosophers problem, `M` would be `PHILOSOPHERS-DINNER` and the following `PHILOSOPHERS-PREDS` module can be its `M-PREDS`:

``` maude
mod PHILOSOPHERS-PREDS is
    protecting PHILOSOPHERS-DINNER .
    including SATISFACTION .

    subsort Table < State .
    op eats : Nat -> Prop [ctor] .

    var  Id : Nat .
    vars L M : List .

    eq < L (ψ | Id | ψ) M > |= eats(Id) = true .
    eq < L > |= eats(Id) = false [owise] .
endm
```

A single parametric proposition `eats` is defined, all whose ground instances are the atomic propositions in the formal sense. The satisfaction relation is equationally defined to make {codemath}`eats($n$)` hold in any state where the $n$-th philosopher has both forks, and so is able to eat. Finally, the module collecting all the specification components is defined, playing the role of `M-CHECK`.

``` maude
mod PHILOSOPHERS-CHECK is
    protecting PHILOSOPHERS-PREDS .        *** atomic propositions
    protecting PHILOSOPHERS-FORMULAE .
    including MODEL-CHECKER .
endm
```

In this module, we will be able to use the model checker with [`check`](#sec:check). We have imported another module, `PHILOSOPHERS-FORMULAE`, that has not been explained yet. Since LTL formulae are represented by terms in Maude, the user can construct and manipulate formulae within the language. For example, we can declare two formulae `someoneEats` and `allEat`, generic on the number of philosophers, to express that at least a philosopher or every philosopher can eat.

```maude
mod PHILOSOPHERS-FORMULAE is
    protecting PHILOSOPHERS-PREDS .
    protecting LTL .

    *** Parameterized formulae for a given number of philosophers
    ops someoneEats allEat : Nat -> Formula .

    var L : List .
    var X Y : Obj .
    var Id N : Nat .

    eq someoneEats(0) = False .
    eq someoneEats(s(N)) = someoneEats(N) \/ eats(N) .

    eq allEat(0) = True .
    eq allEat(s(N)) = allEat(N) /\ <> eats(N) .
endm
```

According to the equations, {codemath}`someoneEats($n$)` is the LTL formula $\bigvee_{k = 0}^{n-1} \mathtt{eats(}k\mathtt{)}$, and {codemath}`allEat($n$)` is $\bigwedge_{k = 0}^{n-1} \diamond \mathtt{eats(}k\mathtt{)}$.

Summing up, the script for preparing a problem to be model checked is:

1.  Specify the model in a system module `M`.
2.  In a protecting extension of `M` including the predefined `SATISFACTION` module (say `M-PREDS`) choose the sort of the model states by making it a subsort of the predefined `State` sort. Declare as many atomic propositions as desired as operators of range `Prop`, and define the satisfaction relation `|=` for all of them.
3.  Write a system module (say `M-CHECK`) combining the specification of the model and properties in `M-PREDS` with the predefined `MODEL-CHECKER` module. Optionally, the `LTL-SIMPLIFIER` module may be included for LTL simplification.

(sec:smodels)=
## Strategy-controlled models

The evolution of strategy-controlled rewriting systems does not only depend on the rules but also on the strategies that limit their application. States in those systems are not univocally associated to terms and their transitions are those rule rewrites allowed by the strategy. We also consider some special options for specific situations, but before describing them we will illustrate the normal behavior with an example.

Remember that unwanted situations may appear in the dining philosophers problem introduced [before](#sec:models). Strategies can prevent some of them by imposing additional restrictions.

```maude
smod PHILOSOPHERS-PARITY is
    protecting PHILOSOPHERS-DINNER .

    strat parity @ Table .
    sd parity := (release
        *** The even take the left fork first
        | (amatchrew L s.t. ψ (o | Id | o) := L /\ 2 divides Id
                  by L using left)
        | left[Id <- 0]
        *** The odd take the right fork first
        | (amatchrew L s.t. (o | Id | o) ψ := L /\ not (2 divides Id)
             by L using right)
        *** When they already have one, they take the other fork
        | (amatchrew L s.t. (ψ | Id | o) ψ := L by L using right)
        | (matchrew M s.t. < L (o | Id | ψ) L' > := M
                 by M using left[Id <- Id])
        ) ? parity : idle .
endsm
```

With the `parity` strategy, even philosophers are compelled to take the left fork before the right one, and the odd should do the opposite. These additional rewriting restrictions produce a different model in which different properties are satisfied. For example, thanks to the `parity` strategy, the situation in which no philosopher can eat is avoided.

Notice that the strategy above is recursive and non-terminating. Even though non-terminating executions cannot be observed with the `srewrite` commands {cite:p}`maude` (§10.4), non-terminating strategies are meaningful and useful to specify the behavior of non-stopping and reactive systems, which are the typical targets of model checking. Non-terminating rewriting paths are not an obstacle for the decidability of model checking as long as they repeat a finite number of states; in other words, as long as they are caused by a loop. The model checker is able to detect cycles also when strategies have parameters.

```maude
smod PHILOSOPHERS-TURNS is
    protecting PHILOSOPHERS-DINNER .

    strat turns @ Table .
    strat turns : Nat Nat @ Table .

    sd turns := matchrew M s.t. < L (o | Id | o) ψ > := M
                      by M using turns(0, s(Id)) .
    sd turns(K, N) := left[Id <- K] ; right[Id <- K] ;
                      release ; turns(s(K) rem N, N) .
endsm
```

The `turns` strategy in the module above makes the philosophers eat in turns. With 3 philosophers, the strategies `turns(0,3)`, `turns(1,3)`, `turns(2,3)` follow continuously in a loop. This is not a convenient solution to the philosophers problem due to the absence of parallelism, but it ensures that all of them eat infinitely often.

Strategies always select a subset of the model executions. Hence, all linear temporal properties satisfied by the unrestricted model will be satisfied by the model controlled by no matter which strategy. However, the model checker also allows considering some explicitly selected strategies as atomic steps, as described hereafter, breaking this rule.

### Finite traces

Strategies are commonly used to specify finite rewriting sequences, those whose results are observed using the `srewrite` commands. Finite executions do not exactly fit in the model-checking setting, where properties are defined for infinite traces. However, they can be assimilated to infinite traces extending their last state forever. Intuitively, the modeled system has stopped after completing its execution and so it will continue in its idle state in perpetuity. Coming back to the philosophers example and considering the recursive strategy `sd free := all ? free : idle .` that coincides with the free execution of the rewriting rules, some finite traces are found:
```text
                                < (o | 0 | o) ψ (o | 1 | o) ψ (o | 2 | o) ψ >
    = right[Id <- 0]}} =>       < (o | 0 | ψ)   (o | 1 | o) ψ (o | 2 | o) ψ >
    = right[Id <- 1]}} =>       < (o | 0 | ψ)   (o | 1 | ψ)   (o | 2 | o) ψ >
    = right[Id <- 2]}} =>       < (o | 0 | ψ)   (o | 1 | ψ)   (o | 2 | ψ)   >
```

This is a finite trace because the strategy `all`, the application of any rule, cannot be executed in the last state. Then, the conditional operator will execute is negative `idle` branch and terminate. The philosophers, unable to do any other movement, will remain like this indefinitely.

The interpretation of finite traces is equivalent to that of terminating or *deadlocked* rewriting sequences in the standard model checker. In fact, the finite execution above is a deadlocked one, but this is not true in general. Suppose `free` were defined `all ; free` instead. Then, the trace above would be discarded since the strategy commits to execute `all` again in the fourth state, which is impossible. Conversely, the strategy `all*` admits all finite rewriting sequences, extended by repeating the last state forever, even if they can be continued otherwise with a genuine rule transition.

### Parallel subterm rewriting

The semantics of a subterm rewriting combinator like the following (or its variants `xmatchrew` and `amatchrew`),

$$\mathtt{matchrew} \; P(X_1, \ldots, X_n) \; \mathtt{by} \; X_1 \; \mathtt{using} \; \alpha_1, \; \ldots, \; X_n \; \mathtt{using} \; \alpha_n$$

is the *parallel* rewriting of its $n$ subterms against their corresponding strategies {cite:p}`theslang`. When looking to the state evolution as a sequence of rewriting steps, this means that the next step may come from any of the subterms. In other words, the execution traces of the `matchrew` are all possible interleavings of all possible combinations of the execution traces of its subterms.

In general, generating all these combinations is computationally expensive. So, an optional and alternative form of *partial order reduction* is offered, which analyzes only those traces where the subterms are rewritten in order, i.e. the rewrites within the subterm $k$ always occur before the rewrites within the subterm $k+1$. It is the user responsibility to ensure by other means that this is enough to prove the correctness of the system.

(sec:opaque)=
### Opaque strategies

The model transitions are fundamentally rule applications, both for the strategy-aware and the standard model checker, but sometimes having strategies as transitions is useful and convenient. We call these strategies *opaque* since the intermediate states occurring during its execution are invisible. Instead, a single transition is seen from the term where the strategy is applied to each of its results. An example of model checking with opaque strategies can be seen [here](https://maude.ucm.es/strategies/modelchecker-manual.pdf) (Section 4.1).

(sec:smodelprep)=
### Model preparation

```{tikz} Structure of the strategy-aware model checker modules.
:align: center

[node distance=.7cm and .8cm, font=\ttfamily]

\node[draw] (SAT) {SATISFACTION};
\node[draw, left=of SAT] (QID) {QID};
\node[draw, left=of QID] (LTL) {LTL};

\node[draw, above=of QID] (BOOL) {BOOL};

\node[draw, below=of QID] (CH) {STRATEGY-MODEL-CHECKER};

\node[draw, right=of SAT] (MP) {M-PREDS};
\node[right=.3cm of MP] (anchor) {\vphantom{M}};
\node[draw, right=.5cm of anchor](SM) {SM};
\node[draw, above=of anchor](M) {M};
\node[draw, below=of anchor] (MC) {SM-CHECK};

\draw[->] (SAT) -- (MP);
\draw[double, ->] (M) -- (MP);
\draw[double, ->] (M) -- (SM);
\draw[double, ->] (MP) -- (MC);
\draw[double, ->] (SM) -- (MC);
\draw[->] (LTL) -- (CH);
\draw[->] (SAT) -- (CH);
\draw[double, ->] (QID) -- (CH);
\draw[->] (CH) -- (MC);
\draw[double, ->] (BOOL) -- (LTL);
\draw[double, ->] (BOOL) -- (QID);
\draw[double, ->] (BOOL) -- (SAT);
```

The problem setting and the usage of the strategy-aware model checker does not differ much of the [standard procedures](#sec:modelprep). The only difference is that, as part of the model specification, a strategy must be provided. The model is now specified by

* a Maude system module `M`, and
* a strategy expression $\alpha$, possible using some strategy definitions in a strategy module `SM` that controls `M`. If the strategy-controlled model checker is used from Maude, the controlling strategy can only be a named strategy without arguments and the strategy module is always required.

For instance, in the dining-philosophers examples, the base rewriting model `M` is the system module `PHILOSOPHERS-PREDS`, and the controlling strategy can be either `parity` in `PHILOSOPHERS-PARITY` or `turns` in `PHILOSOPHERS-TURNS`.

As illustrated in the diagram, the collection of predefined and user-defined modules for strategy-aware model checking is very similar to that for [standard model checking](#sec:modelprep). In the typical setting, the rule-based model is specified in a system module `M`, one or more strategies controlling `M` are defined in an extension of this module that we will call `SM`, its atomic propositions and state are specified in a protected extension of `M` called `M-PREDS` that includes the predefined `SATISFACTION` module, and everything is combined in a strategy module `SM-CHECK`. This latter module imports the predefined `STRATEGY-MODEL-CHECKER` module to allow access to the strategy-aware model checker. The standard `MODEL-CHECKER` module is compatible with the strategy-aware version and both can be imported and used at the same time.

As a final summary, the script for preparing a problem to be model checked with strategies is:

1.  Specify the model in a system module `M`, and possibly define some strategy definitions in a strategy module `SM`.
2.  In a protecting extension of `M`, say `M-PREDS`, choose the sort of the model states by making it a subsort of the `State` sort declared in `SATISFACTION`. Declare as many atomic propositions as desired as operators of range `Prop`, and define the satisfaction relation `|=` for all of them.
3.  Declare a strategy module, say `SM-CHECK`, to combine the model `M` with the property specification in `M-PREDS` and the strategy module `SM` (if any). Import the `STRATEGY-MODEL-CHECKER` module too, and optionally the `LTL-SIMPLIFIER` module for LTL simplification.

(sec:kleene)=
### The Kleene-star semantics of the iteration

The usual meaning of the star operator in formal languages and regular expressions is the Kleene closure, in other words, all finite repetitions of the argument. On the contrary, the iteration operator {codemath}`$\alpha$*` of the strategy language also allows the infinite repetition of $\alpha$, which may be denoted by $\alpha^\omega$. Otherwise, a system controlled by a strategy could not be represented in general by a plain transition system or Kripke structure, since preventing infinite iterations imposes fairness-like restrictions on the infinite behavior of the model that cannot be represented locally. However, these restrictions can be handled by using automata-based algorithms or by pushing them to the temporal formulae being checked. As a method for specifying fairness restrictions directly on the strategy, the `umaudemc` tool supports those approaches {cite:p}`kleene`.

The [`check`](#sec:check) command can be passed a flag `--kleene-iteration` or simply `-k` to make the iteration be interpreted as the Kleene star when checking LTL, CTL, or CTL\* properties (μ-calculus is not supported). If the Spot backend is available, LTL properties will be handled by an extension of the automata-theoretic model-checking approach where the model is a Streett automaton that captures the finiteness restrictions of the iteration. Otherwise, and for all other supported logics, the temporal formulae are extended with premises describing the aforementioned restrictions. Unless there are no iterations, CTL properties become proper CTL\* formulae after the transformation, so a model-checking backend for this more general logic is required and performance could be substantially affected.

The Kleene-star semantics model checker does not directly use the transition system produced by Maude for the strategy-controlled system, but relies on a Python-based implementation of the strategy language included in `umaudemc`, where iterations can be effectively traced. At the moment, the `--full-matchrew` option of the command is always enabled, and the `--merge-states` choice is not properly respected by the LTSmin backend.

(sec:probspec)=
## Probabilistic models

Maude specifications, both standard and strategy-controlled, are intrinsically nondeterministic. However, this nondeterminism can be quantified to yield probabilistic systems that can be analyzed by [probabilistic](#sec:pcheck) and [statistical](#sec:scheck) model-checking methods. The tools described in this documentation offer different alternatives to specify probabilities on top of rewriting systems specified in Maude. In the simplest case, uniform probabilities on the successors are considered for every state. In the most expressive one, probabilities are assigned with arbitrary complex programs in a probabilistic extension of the Maude strategy language. These *probability assignment methods* will be described in this section and illustrated with the following example of a simple coin.

```maude
mod COIN is
    protecting NAT .

    sort Coin .
    ops head tail : -> Coin [ctor] .

    vars C C' L R : Coin .

    rl [thead] : C => head [metadata "8"] .
    rl [ttail] : C => tail [metadata "5"] .

    op inertia : Coin Coin -> Nat .
    eq inertia(C, C') = if C == C' then 2 else 1 fi .
endm
```

Coins can be either in `head` or `tail` state, and they can change from one to the other by the rules `thead` and `ttail`. Other aspects of this module will be explained in due time. The standard rewrite graph of this model is this:

```{tikz}
:align: center

[font=\ttfamily, x=.4\linewidth]

\node[draw, circle] (H) at (0, 0) {head};
\node[draw, circle] (T) at (1, 0) {tail};

\draw[->] (H) edge[bend left=10] node[above] {ttail} (T);
\draw[->] (T) edge[bend left=10] node[below] {thead} (H);
\draw[->] (T) edge[loop right] node[right] {ttail} ();
\draw[->] (H) edge[loop left] node[left] {thead} ();
``` 

The probability assignment methods are `uniform`, `metadata`, `term`, `uaction`, `mdp-uniform`, `mdp-meta­data`, `mdp-term`, and `strategy` for the extension of the strategy language. Some of them derive discrete-time Markov chains (DTMC) out of the rewriting model and others produce Markov decision processes (MDP), which combine nondeterministic and probabilistic behavior. Continuous-time Markov chains (CTMC) can also be derived with the methods `ctmc-uniform`, `ctmc-metadata`, `ctmc-term`, `ctmc-uaction`, and `ctmc-strategy`. All methods except `strategy` are *local*, in the sense that they distribute the probability among the successors of every state separately. Two additional assignment methods, `step` and `pmaude`, are specific to statistical model checking.

### Weight-based assignment methods

All these methods use weights to quantify the likeliness of the successors that are later normalized to obtain probabilities. 

* `uniform` assigns the same probability to every successor of a term, i.e. successors are chosen uniformly at random. For example, the `COIN` module becomes the following DTMC under uniform probabilities:

  ```{tikz}
  :align: center

  [x=.3\linewidth]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle] (T) at (1, 0) {\ttfamily\small tail};

  \draw[->] (H) edge[bend left=10] node[above] {1/2} (T);
  \draw[->] (T) edge[bend left=10] node[below] {1/2} (H);
  \draw[->] (T) edge[loop right] node[right] {1/2} ();
  \draw[->] (H) edge[loop left] node[left] {1/2} ();
  ``` 

* {codemath}`uaction($a_1$=$w_1$, $\ldots$, $a_n$=$w_n$)` receives an assignment of weights to rule labels or actions of the rewriting system. First, the probability is distributed among the labels according to their weights. Then, for each rule label, the successors get an equal share of the assigned probability, in other words, they are then chosen uniformly at random. Instead of weights, actions can be assigned fixed probabilities with {codemath}`$a_k$.p=$p_k$` instead of {codemath}`$a_k$=$w_k$`. Fixed probabilities and weights for different actions can be combined, but fixed probabilities should never sum more than 1. In case no specification is given for an action, a unitary weight is assumed.

  On top of the `COIN` example, the method `uaction(ttail=3, thead=2)` produces the following discrete-time Markov chain:

  ```{tikz}
  :align: center

  [x=.3\linewidth]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle] (T) at (1, 0) {\ttfamily\small tail};

  \draw[->] (H) edge[bend left=10] node[above] {3/5} (T);
  \draw[->] (T) edge[bend left=10] node[below] {2/5} (H);
  \draw[->] (T) edge[loop right] node[right] {3/5} ();
  \draw[->] (H) edge[loop left] node[left] {2/5} ();
  ``` 

  However, since there is a single successor for each action, the uniform distribution of probabilities per action is trivial in this case. The same result is obtained using a fixed probability with the `uaction(ttail.p=0.4)` method.

* `metadata` distributes the probability among the successors according to the weights written in the free-text `metadata` attributes of the rules that caused their transitions. The content of the `metadata` attribute must be a numeric literal or a Maude term of sort `Nat` or `Float` depending on the variables of the rule. A weight of 1 is assumed in case the attribute is missing. Whenever a successor has been generated by multiple rule applications, the weight of one of them is chosen in an implementation-defined way.

  In the rules of the `COIN` module, we can see an example of this kind of specification, where `thead` is assigned a weight of 8 and `ttail` is given a weight of 5. The DTMC produced by this method is the following:

  ```{tikz}
  :align: center

  [x=.3\linewidth]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle] (T) at (1, 0) {\ttfamily\small tail};

  \draw[->] (H) edge[bend left=10] node[above] {5/13} (T);
  \draw[->] (T) edge[bend left=10] node[below] {8/13} (H);
  \draw[->] (T) edge[loop right] node[right] {5/13} ();
  \draw[->] (H) edge[loop left] node[left] {8/13} ();
  ``` 

* {codemath}`term($t$)` is given a Maude term $t$ to calculate weights for every transition on a state and distribute the probabilities according to them. This term must reduce to a literal of numerical sort (`Nat` or `Float`) and it may contain the variables `L`, `R`, and `A` to be instantiated respectively with the left- and right-hand side of the transition, and the label of the rule that caused it, as a term of sort `Qid` with `’unknown` for unlabeled rules. Whenever differently labeled rules produce the same transition, one label is chosen in an implementation-defined way.

  In the `COIN` module, we have already defined a function `inertia` that gives twice as much weight to the current face as to the other one. The equivalent methods `term(inertia(L, R))` and `term(if L == R then 2 else 1 fi)` produce the following probabilistic model:

  ```{tikz}
  :align: center

  [x=.3\linewidth]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle] (T) at (1, 0) {\ttfamily\small tail};
  
  \draw[->] (H) edge[bend left=10] node[above] {1/3} (T);
  \draw[->] (T) edge[bend left=10] node[below] {1/3} (H);
  \draw[->] (T) edge[loop right] node[right] {2/3} ();
  \draw[->] (H) edge[loop left] node[left] {2/3} ();
  ``` 

  Since the variables `L` and `R` have already been declared in the module, we can directly write them in the term. Otherwise, we should have written `L:Coin` and `R:Coin`.

When the local methods are applied in a strategy-controlled model, the procedure does not change except that it is applied on the rewrite graph controlled by the strategy. The `mdp-` variants of the local assignment methods produce a Markov decision process where probabilities are only assigned once the rule label is nondeterministically chosen. There is no `mdp-uaction` method since it would not make any sense. These assignment methods are used for probabilistic model checking in [`pcheck`](#sec:pcheck) and for statistical model checking in [`scheck`](#sec:scheck).

For the `ctmc-` variants of the methods, weights are interpreted as firing rates of the corresponding transitions instead of unnormalized probabilities. In the `ctmc-uniform` method, the firing rate of every transition is unitary. In the case of `ctmc-uaction`, each transition is given the rate specified for its label, but weights are normalized if a fixed probability `.p` is given.

### Strategy-based method

`strategy` uses a probabilistic extension of the strategy language to both control and assign probabilities to the base rewriting system. This method produces a discrete-time Markov chain, a Markov decision process, or even an error depending on how much unquantified nondeterminism is left by the strategy. The new probabilistic combinators of the strategy language are:

  * {codemath}`choice($w_1$ : $\alpha_1$, $\ldots$, $w_n$ : $\alpha_n$)` that selects one of the strategies $\alpha_k$ according to their weights $w_k$. These weights are terms in the `Nat` or `Float` sorts that may contain variables if they are defined in the outer scope. This is an evolution of the nondeterministic choice operator {codemath}`$\alpha_1$ | $\cdots$ | $\alpha_n$`, and similar constructs exist in a probabilistic extension of ELAN {cite:p}`pstratElan` and in Porgy {cite:p}`porgyJournal`.

  * {codemath}`sample $X$ := $\pi$($t_1$, $\ldots$, $t_n$) in $\alpha$` that samples the variable $X$ from a probabilistic distribution with parameters $t_1, \ldots, t_n$ that may contain variables defined in the outer contexts. The new variable $X$ can be freely used in $\alpha$. Both the variable $X$ and the parameters must be of sort `Float`. The available distributions are {codemath}`bernouilli($p$)`, {codemath}`uniform($a$, $b$)`, {codemath}`exp($\lambda$)`, {codemath}`norm($\mu$, $\sigma$)`, and {codemath}`gamma($\alpha$, $\beta$)`. `uniform` can either be the continous or the discrete uniform distribution depending on whether the bounds $a$ and $b$ are of integer type.

    The `sample` operator is not intended for discrete probabilistic model generation when continuous probabilistic distributions are used, but it can generate finite models with discrete distributions. It can be used for statistical model checking in any case.

  * An extension of the `matchrew`, `xmatchrew`, and `amatchrew` combinators of the standard strategy language to allow specifying the weight of every match and select one according to these weights. Syntactically, an optional infix {codemath}`with weight $w$` is added to the original operators, like in

  $$\mathtt{matchrew} \; P(X_1, \ldots, X_n) \; \mathtt{s.t.} \; C \; \mathtt{with \; weight} \; w \; \mathtt{by} \; X_1 \; \mathtt{using} \; \alpha_1, \ldots, X_n \; \mathtt{using} \; \alpha_n$$

  where the weight $w$ is a term of sort `Nat` or `Float` that may contain variables from the matching, the condition, and the outer scope.

For example, the strategy `(choice(2 : ttail, 1 : thead) ; choice(1 : ttail, 3 : thead)) *` specifies the following discrete-time Markov chain if the initial term is `head`:

  ```{tikz}
  :align: center

  [x=.3\linewidth, y=3cm]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle, dashed] (T) at (1, 0) {\ttfamily\small tail};
  \node[draw, circle, dashed] (H2) at (0, -1) {\ttfamily\small head};
  \node[draw, circle] (T2) at (1, -1) {\ttfamily\small tail};
  
  \draw[->] ($(H.west)+(-2.5ex,0)$) -- (H);
  \draw[->] (H) edge[bend left=10] node[above] {2/3} (T);
  \draw[->] (H) edge[bend right=10] node[left] {1/3} (H2);
  \draw[->] (H2) edge[bend right=10] node[right] {3/4} (H);
  \draw[->] (H2) edge[bend left=10] node[above] {1/4} (T2);
  
  \draw[->] (T) edge[bend left=10] node[below] {3/4} (H);
  \draw[->] (T) edge[bend left=10] node[right] {1/4} (T2);
  \draw[->] (T2) edge[bend left=10] node[below] {1/3} (H2);
  \draw[->] (T2) edge[bend left=10] node[left] {2/3} (T);
  ``` 

Notice that the probabilities are now non-local and the graph is made more complex since this strategy has memory. Probabilistic and nondeterministic behavior can be mixed in `(choice(2 : ttail, 1 : thead) ; (ttail | thead)) *` yielding a Markov decision process:

  ```{tikz}
  :align: center

  [x=.3\linewidth, y=3cm]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle, dashed] (T) at (1, 0) {\ttfamily\small tail};
  \node[draw, circle, dashed] (H2) at (0, -1) {\ttfamily\small head};
  \node[draw, circle] (T2) at (1, -1) {\ttfamily\small tail};
  
  \draw[->] ($(H.west)+(-2.5ex,0)$) -- (H);
  \draw[->] (H) edge[bend left=10] node[above] {2/3} (T);
  \draw[->] (H) edge[bend right=10] node[left] {1/3} (H2);
  \draw[->] (H2) edge[bend right=10] node[right] {\texttt{\small thead}, 1} (H);
  \draw[->] (H2) edge[bend left=10] node[above] {\texttt{\small ttail}, 1} (T2);
  
  \draw[->] (T) edge[bend left=10] node[below] {\texttt{\small thead}, 1} (H);
  \draw[->] (T) edge[bend left=10] node[right] {\texttt{\small ttail}, 1} (T2);
  \draw[->] (T2) edge[bend left=10] node[below] {1/3} (H2);
  \draw[->] (T2) edge[bend left=10] node[left] {2/3} (T);
  ``` 

Like for weight-based methods, there is a `ctmc-strategy` that produces a continuous-time Markov chain. The strategy must be free of unquantified nondeterminism, i.e. `strategy` should be able to generate a DTMC.

(sec:probspec-scheck)=
### Statistical-only methods

In addition to the previous general methods, there are some others for statistical model checking only. They can be used with the builtin statistical model checker in [`scheck`](#sec:scheck).


* `step` (with a strategy) considers the given probabilistic strategy as the atomic step of the model, i.e., the $(n+1)$-th state of an execution is a solution of the strategy applied on the $n$-th state. For a sound statistical analysis, this strategy should not contain unquantified nondeterminism. In other words, it should provide a single (random) solution in every execution. If the strategy does not provide any solution, the execution continues in the same state.

  For example, the `step` method with the strategy `choice(2 : ttail, 1 : thead)` yields the following discrete-time Markov chain, even though it will not be explicitly constructed.

  ```{tikz}
  :align: center

  [x=.3\linewidth, y=3cm]

  \node[draw, circle] (H) at (0, 0) {\ttfamily\small head};
  \node[draw, circle, dashed] (T) at (1, 0) {\ttfamily\small tail};
  \node[draw, circle, dashed] (H2) at (0, -1) {\ttfamily\small head};
  \node[draw, circle] (T2) at (1, -1) {\ttfamily\small tail};
  
  \draw[->] ($(H.west)+(-2.5ex,0)$) -- (H);
  \draw[->] (H) edge[bend left=10] node[above] {2/3} (T);
  \draw[->] (H) edge[bend right=10] node[left] {1/3} (H2);
  \draw[->] (H2) edge[bend right=10] node[right] {\texttt{\small thead}, 1} (H);
  \draw[->] (H2) edge[bend left=10] node[above] {\texttt{\small ttail}, 1} (T2);
  
  \draw[->] (T) edge[bend left=10] node[below] {\texttt{\small thead}, 1} (H);
  \draw[->] (T) edge[bend left=10] node[right] {\texttt{\small ttail}, 1} (T2);
  \draw[->] (T2) edge[bend left=10] node[below] {1/3} (H2);
  \draw[->] (T2) edge[bend left=10] node[left] {2/3} (T);
  ``` 

  Sample operators can also be used.

* `step` (without a strategy) consider a rule application as the atomic step of the model. For the statistical analyses to be sound, there must be a single possible (random) rewrite at every step.

* `pmaude` assigns probabilities to an APMaude model according to the conventions of that framework {cite:p}`pmaude`. In particular, the Maude specification must define a function `tick` on the state sort and a function `getTime` from the state sort to `Float`. Every execution $(t_n)_{n=0}^\infty$ starts by rewriting the initial term with the rules of the module to obtain $t_0$, and then $t_{n+1}$ is obtained by rewriting {codemath}`tick($t_n$)` exhaustively with the rules of the module. Randomness relies on the PMaude infrastructure and the `random` and `counter` operators in the Maude prelude, and random seeds are refreshed on each execution. If a strategy expression is provided, it will be ignored.
