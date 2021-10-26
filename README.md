Unified Maude model-checking tool
=================================

Uniform interface to different model checkers for standard and
strategy-controlled Maude models.

In addition, the tool offers a graphical interface to the model checkers,
allows generating graphs of the models, running test suites and
benchmarking them. This functionality is organized in subcommands:

* `check <filename> <initial term> <formula> [<strategy>]` to check a
temporal property on the given rewrite system.
* `pcheck <filename> <initial term> <formula> [<strategy>] [--assign <method>] [--reward <term>]` to calculate probabilities and expected values for a temporal property on the given rewrite system extended with probabilities.
* `graph <filename> <initial term> [<strategy>]` to generate a visual
representation of the reachable rewrite graph from the given initial state
in the [GraphViz](https://graphviz.org/)'s DOT format.
* `gui [--web]` to model check from a graphical user interface.
* `test [--benchmark] <filename>` to test or benchmark model-checking cases.

Formulae are expressed in a syntax extending the `LTL` module of the 
`model-checker.maude` file in the official distribution. For CTL and CTL*, the
universal `A_` and existential `E_` path quantifiers are available. For the
μ-calculus, there are the universal `[_]_` and existential `<_>_` modalities,
whose first argument is a space-separated list of rule labels, and the
fixpoint operators `mu_._` and `nu_._`. The argument of the modalities may also
be a dot to mean that any action is fine, and the list may be preceded by a
negation symbol `~` to indicate its complement.

Probabilistic model checking is available through the `pcheck` command with a
syntax similar to `check`. By default, it calculates the probability of the 
given LTL, CTL, or PCTL formula. For a reachability formula, the options
`--steps` and `--reward <term>` can be passed to calculate instead the expected
number of steps or reward, respectively. Formulae
admit the `P` operator of PCTL and the TCTL-like bounds for LTL operators, as
specified [here](umaudemc/data/problog.maude). Probabilities are assigned with
some alternative methods explained [below](#specification-of-probabilities).


Dependencies
------------

The `umaudemc` tool requires the [`maude`](https://pypi.org/project/maude) package
and Python 3.7 or newer. This provides support for LTL, CTL and μ-calculus
model checking. However, other external backends can be installed:

* The built-in LTL model checker included in Maude and its extension for
strategy-controlled systems are already available in the `maude` library.
* [LTSmin](https://ltsmin.utwente.nl) is a language-independent model checker,
for which a [Maude language extension](http://maude.ucm.es/strategies)
has been written. The environment variable `LTSMIN_PATH` should be set to
the path containing the LTSmin binaries and `MAUDEMC_PATH` should point
to the full path of the language plugin. Since the official version of LTSmin
does not support mixing edge labels and state labels in μ-calculus formulae,
a ready-to-use distribution of a
[modified version](https://github.com/fadoss/ltsmin) can be downloaded from
[here](http://maude.ucm.es/strategies).
* [pyModelChecking](https://pypi.org/project/pyModelChecking/) is a simple
Python model-checking library. It can be installed with
`pip install pyModelChecking`.
* [NuSMV](http://nusmv.fbk.eu/). The environment variable should be set to
the path where the `NuSMV` binary is available (if not already in the system
path).
* [Spot](https://spot.lrde.epita.fr/) is a platform for LTL and ω-automata
manipulation. Its Python library should be installed as explained in its
webpage.
* The `umaudemc` tool includes a built-in μ-calculus implementation based on
the procedure described [here](https://doi.org/10.1007/978-3-319-10575-8_26)
and Zielonka's algorithm.

The following table shows the temporal logics supported by each of them:

| Logic      | `maude`     | `ltsmin`    | `pymc`   | `nusmv`  | `spot`   | `builtin` |
| ---------- | ----------- | ----------- | -------- | -------- | -------- | --------- |
| LTL        | on-the-fly  | on-the-fly  | tableau  | tableau  | automata |           |
| CTL        |             | ✓           | ✓        | ✓        |          | ✓         |
| CTL*       |             | ✓           | ✓        |          |          |           |
| μ-calculus |             | ✓           |          |          |          | ✓         |

The first available and compatible backend in the order above will be used to
model check the given formula. The default order can be overwritten using the
`--backend` argument followed by a comma-separated list of backend names as
they appear in the table.

For the probabilistic model-checking command `pcheck`, two backends are supported:
* [PRISM](https://www.prismmodelchecker.org/). If not installed in the system path,
its location should be provided using the `PRISM_PATH` environment variable.
* [Storm](https://www.stormchecker.org/). Its location should be specified
with the `STORM_PATH` environment variable if not available in the system path.

Moreover, to read test cases specifications in YAML, the 
[PyYAML](https://pypi.org/project/pyaml/) package is required.


Documentation
-------------

The available program options are shown when the `--help` is introduced.
Some deserve more explanations:

* `--slabel` sets the format of state labels that appear in counterexamples and
graphs. The format string may contain the templates `%t`, `%s`, `%i` that will
be replaced by the subject term, the next strategy to be executed (only in
strategy-controlled systems), and the internal index of the state,
respectively. An optional length limit can be written between the `%` and the
letter. Text between curly brackets will be interpreted as arbitrary Maude
terms and reduced in the current module, after replacing the `%t` templates in
it. This is useful to make traces and graphs more readable, and to check atomic
propositions like in `%t -- p={%t |= p}`.
* `--elabel` sets the format of edge labels in counterexamples and graphs. The
format string may contain the templates `%s`, `%l`, `%o` that will be replaced
by the statement the caused the transition, by its label, and by the string
`opaque` if the transition was caused by an opaque strategy, respectively.
* The model adaptations for branching-time logics `--purge-fails` and
`--merge-states` are chosen automatically by the tool depending on the input
formulae. However, they can be manually overwritten.
* The option `--kleene-iteration` or `-k` in `check` when checking properties
on strategy-controlled specifications makes the iteration strategy be
interpreted as the Kleene star (i.e. infinite iterations are discarded).

### Specification of probabilities

For the probabilistic model-checking command `pcheck`, Maude specifications
should be extended with probabilistic information. Several alternative methods
allow assigning probabilities to the successors of each state through the
`--assign` argument:

* `uniform` assigns the same probability to every successor. This is the default
if no `--assing` option is provided.
* `uaction(a1=w1, ..., an=wn)` specifies weights for the actions (rule labels)
of the model (omitted actions take a unitary weight), which are used to
distribute the probability among them. If there are multiple successors for the
same action, probabilities are shared uniformly among these successors. A fixed probability can also be assigned to an action with `a.p=x` instead of `a=w` (of course, they cannot sum more than 1).
* `metadata` uses the `metadata` attribute of the rules in Maude to specify
weights for each transition. In case of multiple successors produced by the same
rule, probability is distributed uniformly among them. 
* `term(t)` evaluates a term on every transition of the model to calculate its
weights. The term `t` may contain a variable `L` for the origin of the transition,
`R` for the result, and `A` of sort `Qid` for the action name.
* `strategy` extracts the probabilities of the model from a probabilistic strategy expression containing `choice` operators. This allows controlling
rewriting and specifying probabilities at the same time.

All methods from `uniform` to `term` produce discrete-time Markov chains (DTMC) from the input rewrite system. Their variants `mdp-uniform`, `mdp-metadata`, and `mdp-term`
can be used instead to generate Markov decision processes (MDP) and calculate minimum
and maximum probabilities. The `strategy` method may produce a DTMC if all nondeterministic options have been quantified, an MDP if nondeterministic choices precede quantified ones between rewrites, and fails with an error message otherwise.


References
----------

More details about the integration of the external classical model checkers can be found
in [*Strategies, model checking and branching-time properties in Maude*](https://doi.org/10.1016/j.jlamp.2021.100700).
