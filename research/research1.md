# research1 — QRE primer and the one-price objection (plain-text conversion)

> Converted from `research1.rtf` (the original problem statement). Formatting (tables,
> maths) is degraded by conversion; the RTF remains the authoritative original.


If you already know Nash equilibrium, mixed strategies, logit choice models, and competitive pricing, the cleanest way to understand Quantal Response Equilibrium (QRE) is:
QRE replaces the assumption of perfectly optimal best responses with probabilistic responses that become increasingly concentrated on higher-payoff actions as rationality increases.
So instead of asking “Which price is my exact best response?”, QRE asks:
“Given my beliefs about competitors' prices, what probability do I assign to each price, considering that higher-payoff prices are more likely to be chosen?”
1. Start from ordinary Nash
Suppose two petrol stations choose prices pi and pj .
Your expected profit from choosing pi is
Ui (pi ,pj )
and under Nash, player i chooses
pi∗ ∈argpi max Ui (pi ,pj∗ )
The crucial assumption is deterministic best response:
P(pi∗ )=1
and
P(pi =pi∗ )=0.
This is appropriate if agents are perfectly rational and there are no decision errors.

2. QRE makes the response probabilistic
QRE says that even if one price generates a higher expected profit, other prices can still be selected.
The canonical logit QRE is
Pi (p)=∑p′ exp(λEUi (p′))exp(λEUi (p))
where:
	•	EUi (p) = expected payoff from price p
	•	λ = rationality / precision parameter
	•	Pi (p) = probability of choosing price p
This is essentially a softmax best response.
The key parameter is λ
If
λ=0
then
Pi (p)=∣P∣1
so every price is equally likely.
As
λ→∞
the distribution converges toward the ordinary best response:
Pi (p∗)→1.
So you can think of QRE as creating a continuum:
random behaviour⟶boundedly rational behaviour⟶perfect rationality
as λ increases.

3. The subtle part: equilibrium
The interesting thing isn't the softmax itself.
The interesting thing is that everyone is responding probabilistically to everyone else's probabilistic behaviour.
For two firms:
σ1 (p)=∑p′ exp[λ1 EU1 (p′,σ2 )]exp[λ1 EU1 (p,σ2 )]
and
σ2 (p)=∑p′ exp[λ2 EU2 (p′,σ1 )]exp[λ2 EU2 (p,σ1 )] .
A QRE is a fixed point:
σ1 =BR1QRE (σ2 ) σ2 =BR2QRE (σ1 ).
That's the conceptual leap.
In Nash:
"I know your strategy, so I choose my best response."
In QRE:
"I know your probability distribution over strategies, so I choose my own probability distribution according to a noisy/quantally rational response."

4. Why this is particularly interesting for pricing
Imagine two petrol stations with possible prices:
{1.70,1.72,1.74,1.76,1.78}.
Suppose, given Station B's price distribution, Station A's expected profits are:
Price|Expected profit|
£1.70|102|
£1.72|108|
£1.74|115|
£1.76|113|
£1.78|104|
Nash says:
pA =1.74.
QRE doesn't.
With, say,
λ=0.1,
you calculate
P(1.70)∝e0.1(102) P(1.72)∝e0.1(108)
etc.
After normalization, £1.74 has the highest probability, but £1.76 and even £1.72 still have meaningful probabilities.
That is often much closer to observed pricing behaviour.

5. The really important distinction from ordinary mixed Nash
This is where QRE becomes more interesting to someone who already knows game theory.
A mixed-strategy Nash equilibrium doesn't mean:
"Players make random mistakes."
It means players are indifferent among the actions in the support.
For example, if
P(A)=0.6,P(B)=0.4
in a mixed Nash equilibrium, the supported strategies have equal expected payoff.
QRE doesn't require that.
You can have
EU(A)=10
and
EU(B)=8
yet still have
P(B)>0.
The probability difference reflects payoff sensitivity, rather than indifference.
That's why QRE is often described as a model of bounded rationality / stochastic choice, rather than merely another equilibrium refinement.

6. QRE gives you something Nash doesn't
Suppose the payoff difference between two prices is tiny:
EU(£1.74)=100 EU(£1.75)=99.8.
A deterministic Nash model says:
£1.74 wins. Choose it.
But empirically, it would be unreasonable to expect firms to behave as though a 20p-per-day difference is infinitely important.
QRE says:
P(£1.74)>P(£1.75)
but perhaps only slightly.
That's economically meaningful.
The magnitude of the payoff difference matters.

7. Interpretation of λ
This is where I would be careful with the term rationality.
Mathematically, λ is a precision parameter.
Consider two prices with expected profits:
U1 =100,U2 =95.
Then
P2 P1 =eλ(U1 −U2 )=e5λ.
Therefore:
Low λ
λ=0.1
gives
P2 P1 =e0.5≈1.65.
Only modest preference.
High λ
λ=1
gives
P2 P1 =e5≈148.
Almost deterministic.
So λ controls how strongly payoff differences translate into choice probabilities.

8. Connection to discrete choice
Given your pricing background, this is probably the most useful mental model:
Logit demand
You may already have something like
P(j)=∑k eVk eVj
where consumers choose products probabilistically according to utility.
QRE essentially applies the same mathematical machinery to strategic players.
Instead of:
Vj =consumer utility
you have:
Vj =EUi (j,σ−i )
—the expected payoff to the strategic player.
So:
QRE is essentially a logit response model embedded inside a game.
That is one of the easiest ways to conceptualize it if you're already comfortable with discrete-choice pricing models.

9. And this produces an interesting feedback loop
For pricing, you get:
Competitor price distribution→your expected profits→your price probabilities→competitor’s expected profits→their price probabilities
until the system reaches a fixed point.
For example:
σB →EUA →σA →EUB →σB .
At convergence, nobody has an incentive to change their choice distribution, given the other firm's distribution.
That's QRE.

10. QRE vs Nash in one picture
You can think of the best-response function changing from:

Nash

probability
1 |             █
  |             █
  |             █
0 | █ █ █ █ █ █ █
    prices

to something like:

QRE

probability
  |
  |             █
  |           █ █ █
  |         █ █ █ █ █
  |       █ █ █ █ █ █
  +--------------------
       prices

The best price still gets the largest probability, but nearby alternatives aren't eliminated.

11. Why QRE can be useful in your pricing work
For something like petrol/gas-station pricing, QRE gives you a particularly nice framework because firms may:
	•	observe competitor prices imperfectly;
	•	have delayed information;
	•	have heterogeneous pricing algorithms;
	•	react imperfectly to small margin differences;
	•	have operational constraints;
	•	use heuristics;
	•	anticipate competitor reactions imperfectly.
Rather than explicitly modelling every source of imperfection, QRE gives you a parsimonious behavioural parameter:
λ.
You can then ask:
What happens to equilibrium pricing as firms become more strategically precise?
For example:
λ=0.1→λ=0.5→λ=1→λ=5
and observe how the equilibrium price distributions sharpen.

12. One particularly important insight
QRE isn't simply:
"Nash equilibrium with randomness."
The deeper idea is:
The equilibrium object itself is a probability distribution over actions, generated by strategically consistent stochastic responses.
Nash asks:
Which actions are optimal?
QRE asks:
How does the probability of choosing an action depend on how optimal it is, when everyone behaves this way?
That makes QRE especially attractive when your empirical object is a distribution of prices, rather than a single equilibrium price.
And that's probably the most useful way to think about it for a pricing model such as your ThermoQRE work: instead of predicting "the equilibrium price is £1.76", the model can predict something like "£1.76 has a 37% probability, £1.75 has 28%, £1.77 has 21%,...", with the distribution emerging endogenously from strategic interaction.



In reality you still have to choose one price...so what's the use of a probability...you them tend to increase your lambda and land on the same result as Nash...debate on this

Yes — that is a legitimate criticism of QRE, and it gets at one of the most important conceptual issues with using QRE for real-world pricing.
Your argument is essentially:
"The firm doesn't submit a probability distribution to the market. It has to submit one price. So why should I care about a QRE distribution? If I eventually need to select one price, why not just increase λ until QRE collapses onto the Nash best response?"
I think that's a strong argument if QRE is being used as a prescriptive pricing algorithm. But it is much weaker if QRE is being used as a positive behavioural model.
There are really two different uses of QRE.

1. Your criticism is exactly right for "what price should I charge?"
Suppose QRE gives:
Price|QRE probability|
£1.72|8%|
£1.73|15%|
£1.74|27%|
£1.75|31%|
£1.76|14%|
£1.77|5%|
But tomorrow you must actually set:
p∈{1.72,…,1.77}.
What do you do?
You could choose:
	•	mode → £1.75
	•	expected price → perhaps £1.745
	•	randomly sample → potentially any of them
	•	maximize expected profit again → back to best response
And that last option is the problem you identified.
If your objective is simply:
pmax E[π(p)]
then the QRE probabilities aren't necessarily adding anything to the decision rule.
You could simply calculate expected profit for every price and select:
p∗=argpmax E[π(p)].
That's basically the Nash-style best response.
So QRE should not automatically be sold as a better pricing optimiser than Nash.

2. But there's a subtle mistake in saying "therefore QRE is useless"
The probability isn't necessarily a prescription for what you should do.
It can instead describe how firms actually behave.
That's a completely different question.
Imagine observing 10,000 daily prices from petrol stations.
You find:
When £1.74 is estimated to be the profit-maximising price, firms choose it 61% of the time, £1.73 18%, £1.75 14%, and other prices 7%.
A Nash model predicts:
P(1.74)=1.
That's empirically terrible.
QRE can say:
P(p)∝eλEU(p).
Now the distribution itself is the object you are trying to explain.
You're no longer saying:
"The firm should randomise between these prices."
You're saying:
"Given strategic incentives, these are the empirically plausible probabilities with which firms select different prices."
That's a much stronger use of QRE.

3. Think of it as analogous to discrete-choice demand
This is probably the most useful analogy for your pricing background.
Suppose a consumer has utilities:
UA =10.0 UB =9.8.
You don't say:
"The consumer has to buy one product, therefore the probability model is useless."
The consumer does buy one product.
The probability model is useful because you're modelling the distribution of choices across repeated decisions / heterogeneous consumers / unobserved shocks.
QRE is doing something analogous on the strategic supply side.
The petrol station ultimately chooses one price.
But the model says that conditional on its information, incentives, noise, heterogeneity, bounded rationality, etc., the probability of selecting each feasible action is:
P(pi ).
The realised price is one draw from that behavioural process.

4. But this exposes a deeper question: where does the randomness actually come from?
This is where I'd push QRE much harder than the standard textbook presentation.
If you tell me:
"The firm chooses £1.74 with 40% probability and £1.75 with 35% probability."
I'd immediately ask:
Probability according to what?
Is the firm literally randomising?
Usually, no.
There may instead be:
	•	private information;
	•	unobserved demand shocks;
	•	manager-specific preferences;
	•	imperfect competitor-price observation;
	•	different forecasts;
	•	different operational constraints;
	•	algorithmic noise;
	•	different beliefs about elasticity;
	•	integer/price-grid constraints;
	•	adjustment costs;
	•	inertia;
	•	organisational behaviour.
QRE compresses all of this into the stochastic response.
That's useful—but also potentially dangerous.
Because λ can become a garbage parameter absorbing everything you didn't model explicitly.

5. And this is where your "just increase λ" argument becomes powerful
Suppose:
P(pi )=∑j exp(λEU(pj ))exp(λEU(pi )) .
As:
λ→∞
you get:
P(p∗)→1.
So yes:
QREλ→∞ Nash
in the appropriate sense.
Therefore, if your only criterion is profit maximisation under a known payoff model, QRE doesn't magically dominate Nash.
It actually contains Nash as its limiting case.

6. But there's an even more interesting problem
Increasing λ isn't necessarily "becoming more rational."
That's an interpretation we often attach to the parameter.
Mathematically, you're saying:
"Small payoff differences have a weak effect on choice."
versus
"Small payoff differences have a strong effect on choice."
Consider:
EU(p1 )=100.0 EU(p2 )=99.9.
At high λ, you effectively say:
"A 10p difference makes the firm overwhelmingly choose p1 ."
But a real pricing organisation might have a much bigger source of uncertainty than 10p of expected profit.
So high λ isn't necessarily more realistic.
It is only more consistent with the assumption that firms respond very precisely to payoff differences.

7. There's also an important distinction between QRE and "randomising"
This is often misunderstood.
A QRE does not necessarily mean the manager wakes up every morning and flips a weighted coin.
You can interpret the logit formulation as arising from a latent payoff:
U~i (p)=EUi (p)+ϵip
where ϵip represents unobserved payoff/decision noise.
The firm chooses:
pi =argpmax [EUi (p)+ϵip ].
If the ϵ's have the appropriate extreme-value distribution, you get the logit QRE probabilities.
Now the probability has a much more natural interpretation.
The firm still chooses one price.
We—the modeller—don't observe the firm's complete latent decision environment.
Hence we predict a probability distribution.
That is much more defensible.

8. Now let's turn your argument around
You said:
"You then tend to increase λ and land on the same result as Nash."
I'd modify that slightly.
If you are doing optimisation:
Yes.
You should probably not use QRE merely to generate a probability and then pick a price. Just solve the optimisation problem.
If you are doing prediction:
No.
You should estimate λ from observed behaviour rather than arbitrarily increasing it.
For example:
λ^=argλmax t∑ logPλ (pt ∣competitor behaviourt ).
Now λ is estimated from actual pricing decisions.
If the estimated λ is low, that's evidence of substantial stochasticity / imperfect response.
If it's very high, the data are telling you behaviour is close to best response.
That's an empirical question rather than a modelling preference.

9. This gives QRE a very different role in a pricing system
I'd divide the approaches like this:
Question|Nash|QRE|
What is the theoretically optimal price?|Excellent|Not primary purpose|
Find equilibrium strategy|Excellent|Yes|
Predict one firm's price|Can be brittle|Potentially better|
Explain price dispersion|Poor|Good|
Model imperfect response|Poor|Good|
Estimate behavioural rationality|No|Yes|
Forecast distribution of prices|Poor|Good|
Optimise tomorrow's price|Excellent|Not necessarily|
Capture heterogeneous behaviour|Limited|Potentially good|
So I wouldn't claim:
QRE is better than Nash for pricing.
I'd claim:
QRE answers a different question from Nash.

10. And there's a major issue for your ThermoQRE idea
This actually suggests a better architecture.
Don't make QRE the decision rule.
Make it the strategic environment model.
For example:
Layer 1 — Strategic model
Estimate:
P(pj ∣state)
for competitors using QRE.
This gives you a probability distribution over competitor prices.
Layer 2 — Your optimisation
Then calculate:
E[πi (pi )]=pj ∑ P(pj )πi (pi ,pj ).
Layer 3 — Choose your actual price
Finally:
pi∗ =argpi max E[πi (pi )].
Now QRE is doing something genuinely useful.
It isn't telling you to randomise your price.
It is telling you:
"Don't assume your competitor will deterministically play their Nash price. Here's the distribution of competitor behaviour you should optimise against."
That is a substantially stronger use case.

11. And there's one more twist: QRE can change your best response even when you don't randomise
Suppose your competitor has:
P(pB )=⎩⎨⎧ 0.20.50.3 £1.70£1.75£1.80
You can calculate your expected profit for each of your own prices.
You might discover:
EU(1.74)=£1,020 EU(1.75)=£1,018 EU(1.76)=£1,015.
You still choose £1.74 deterministically.
But your calculation is based on a stochastic model of the competitor.
That's where QRE can matter without requiring you to randomise.

So my verdict on your challenge
I'd actually agree with about 70% of your criticism.
The weak argument for QRE is:
"Real firms aren't perfectly rational, therefore give them probabilities."
That's not enough.
And:
"We have to choose one price, so let's take the highest-probability QRE price."
is also not a particularly compelling decision theory.
But the stronger argument is:
Nash is a theory of strategic equilibrium under exact best responses. QRE is a behavioural model of how observed actions depart from those best responses.
And in pricing, the really valuable thing may not be the probability that you charge £1.75.
It may be the probability that your competitor charges £1.74, £1.75, £1.76, etc.—because your optimal deterministic price can then be calculated against that distribution.
That distinction turns QRE from "why would I randomise my price?" into a much more defensible stochastic model of the competitive environment.
And there is a genuinely interesting research question lurking here: whether a QRE-based stochastic competitor model actually produces better out-of-sample pricing predictions than Nash, mixed Nash, or a conventional demand/response model. That's where I'd test QRE rather than arguing about it philosophically.



Where does this leave demand forecasting/elasticity modelling which is the pricing world's linchpin...qre (and also nash) do they account for it implicitly (likely)...and isn't qre similar to max entropy...just different math

Yes — and I think this is where the distinction becomes much more interesting. There are three layers that are easy to conflate:
	1	Demand model / elasticity — what happens to quantity when prices change?
	2	Strategic model (Nash/QRE) — how do competitors account for each other's actions?
	3	Choice/noise model (QRE / entropy) — how precisely do they translate incentives into an action?
QRE does not replace demand forecasting. In a serious pricing system, I'd put demand/elasticity underneath the game-theoretic layer.
1. Nash doesn't inherently contain demand elasticity
Suppose your firm's profit is
πi (pi ,pj )=(pi −ci )Qi (pi ,pj ).
The Nash equilibrium is defined by
pi∗ ∈argpi max Epj [πi (pi ,pj )].
But where did Qi come from?
That's your demand model.
It might be:
Qi =f(pi ,pj ,X)
where X includes seasonality, weather, location, promotions, competitor characteristics, etc.
So elasticity is absolutely central.
For example, with a simple demand curve
Q(p)=Ap−ϵ,
the firm's optimal price depends directly on ϵ.
In the simplest monopoly case:
pp−c =ϵ1 .
So Nash isn't somehow implicitly discovering elasticity. You supply the payoff function, and elasticity is embedded in that payoff function.



Ed = %ΔP%ΔQd
T
R
=
P
×
Q
d
=
50
×
50
=
2
,
500
TR = P \times Q_d = 50 \times 50 = 2,500
TR=P×Qd =50×50=2,500
E
d
E_d
Ed
0.50

E
d
E_d
Ed
P
P
P
50.0

P
P
P

20
40
60
80
100
20
40
60
80
Q
Q
Q
P
P
P
TR = 2,500
Give feedback

2. In a competitive pricing game, elasticity gets more interesting
Suppose demand is
Qi =Qi (pi ,pj ).
Then the firm's first-order condition is something like
∂pi ∂πi =Qi +(pi −ci )∂pi ∂Qi =0.
Now the relevant object isn't just own-price elasticity.
It's the strategic demand system:
	•	own-price elasticity
	•	cross-price elasticity
	•	competitor-price response
	•	substitution patterns
	•	market share
	•	potentially dynamic effects.
So a Nash pricing model is really:
Demand model+strategic interaction→equilibrium prices
That's how I'd conceptualise it.

3. QRE sits one level above that
Now replace the deterministic best response with:
Pi (p)∝exp{λEUi (p)}.
But notice what is inside EUi (p):
EUi (p)=pj ∑ Pj (pj )determined by demand modelπi (p,pj ) .
And:
πi =(p−c)Qi (p,pj ).
So the architecture is:
Demand/elasticity→profit→strategic expected profit→QRE response
Therefore your intuition is correct:
QRE does account for elasticity, but only because elasticity is already embedded in the payoff function.
QRE itself doesn't estimate elasticity.

4. This distinction is extremely important
Imagine your demand model says:
ϵ=−4.
QRE doesn't tell you that.
And if you change your demand model to:
ϵ=−1.5,
QRE doesn't magically adjust its understanding of consumers.
The resulting QRE changes because the payoffs change.
That's quite different.
So I'd resist saying:
"QRE implicitly models demand."
Better:
"QRE propagates the consequences of the demand model through strategic behaviour."
That's a much more accurate statement.

5. Now your Max Entropy point is really interesting
Yes, there is a deep connection.
In fact, the logit QRE response can be derived as the solution to an entropy-regularised optimisation problem.
Instead of:
σi max Eσi [Ui ]
you solve:
σi max [Eσi [Ui ]+λ1 H(σi )]
where
H(σi )=−a∑ σi (a)lnσi (a)
is Shannon entropy.
The solution is:
σi (a)=∑b exp(λUi (b))exp(λUi (a)) .
That's precisely the logit response.
So your intuition is not merely an analogy.
There is an actual mathematical equivalence.

6. But I wouldn't say "QRE is just Max Entropy"
I'd say:
Logit QRE can be interpreted as maximum-entropy / entropy-regularised choice subject to expected-payoff considerations.
The distinction matters because the game is what makes QRE different.
Maximum entropy by itself asks something like:
Given constraints/information, what probability distribution has maximum entropy?
QRE asks:
Given that other players are themselves strategically responding, what probability distribution is the entropy-regularised optimal response?
So:
MaxEnt→probability distribution
whereas
QRE=strategic game+stochastic/entropy-regularised response

7. And there's a beautiful connection to statistical mechanics
This is where the mathematics becomes surprisingly elegant.
The QRE distribution:
P(a)=ZeλU(a)
looks exactly like a Boltzmann/Gibbs distribution:
P(x)=Ze−βE(x) .
The correspondence is approximately:
Statistical mechanics|QRE|
Energy E|Negative payoff −U|
Inverse temperature β|Rationality λ|
Temperature T|1/λ|
Boltzmann distribution|Logit response|
Ground state|Best response|
High temperature|Noisy/random behaviour|
Low temperature|Near-deterministic behaviour|
So your earlier intuition about QRE being related to maximum entropy is pointing toward a very real statistical-mechanical interpretation.

8. But there's a critical pricing implication
And this comes back to your original objection.
If I give you:
Q(p)=f(p)
and you have a good estimate of elasticity, the demand model is usually doing far more economic work than QRE.
Suppose you're deciding whether to move petrol price from £1.70 to £1.72.
The critical question is:
How many litres do I lose?
That's elasticity.
QRE doesn't answer that.
It answers:
Given the resulting profit consequences and competitor behaviour, how precisely does the firm respond to those incentives?
So in the hierarchy:
Demand>strategic interaction>choice stochasticity
for many practical pricing applications.
Not universally—but that's a good default.

9. This also exposes a potential weakness in using QRE for your ThermoQRE model
If you build:
QRE→price
without a serious demand model underneath it, you've essentially built a game-theoretic pricing toy model.
But if you build:
Demand forecasting→elasticity→profit surface→competitor response→QRE→optimal price
then you're doing something much more interesting.
And you can actually ask an empirical question:
Does modelling competitor behaviour as QRE improve pricing decisions beyond a conventional elasticity-based optimisation model?
That's the experiment I'd run.

10. There's an even deeper possibility
You could potentially make λ state-dependent.
Instead of:
λ=constant,
have:
λt =f(market volatility,competitor uncertainty,price dispersion,time of day,information quality,…).
Then QRE becomes a model where strategic precision changes with the environment.
For example, firms might behave approximately Nash-like when competitor prices are stable:
λ→high
but become much more diffuse when the market is volatile:
λ→low.
Now you're getting something that standard Nash can't represent.

The conceptual stack I'd use
If we strip away the terminology, I think the cleanest architecture is:

                 CUSTOMER BEHAVIOUR
                        │
                        ▼
              Demand / Elasticity
                        │
                        ▼
                  Profit surface
                        │
                        ▼
              Strategic interaction
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
             Nash                QRE
        exact response       noisy/entropy-
                             regularised response
              │                   │
              └─────────┬─────────┘
                        ▼
                  Price decision

And MaxEnt sits underneath the QRE response mechanism, not as an alternative to demand modelling.
So I'd sharpen the earlier debate considerably:
Demand modelling tells you what the economic consequences of a price are. Nash tells you how a perfectly strategic firm responds to those consequences. QRE tells you how a strategically interacting firm responds when the mapping from payoff to action is probabilistic/entropy-regularised.
And your observation that logit QRE is essentially an entropy-regularised best response is exactly the bridge between the game-theory and MaxEnt perspectives.


