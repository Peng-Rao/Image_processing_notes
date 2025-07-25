#import "@preview/ctheorems:1.1.3": *
#let definition = thmbox("definition", "Definition", inset: (x: 0em, top: 0em))
#let proposition = thmbox("proposition", "Proposition", inset: (x: 0em, top: 0em))
#let theorem = thmbox("theorem", "Theorem", inset: (x: 0em, top: 0em))
#let lemma = thmbox("lemma", "Lemma", inset: (x: 0em, top: 0em))
#let example = thmbox("example", "Example", inset: (x: 0em, top: 0em))
#let trace = $op("trace")$
#let rank = $op("rank")$
#let prox = $op("prox")$
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$
#let shrink = $op("shrink")$
#let Var = $op("Var")$
#let sign = $op("sign")$
#let diag = $op("diag")$
#let MSE = $op("MSE")$
#let Bias = $op("Bias")$
#let RR = $bb(R)$
#let EE = $bb(E)$
#let rank = $op("rank")$

#let nonum(eq) = math.equation(block: true, numbering: none, eq)
#let firebrick(body) = text(fill: rgb("#b22222"), body)
