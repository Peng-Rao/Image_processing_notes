#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#show: simple-note.with(
  title: [
    Mathematical Models and Methods For Image Processing
  ],
  date: datetime(year: 2025, month: 2, day: 17),
  authors: (
    (
      name: "Rao",
      github: "https://github.com/Peng-Rao",
      homepage: "https://github.com/Peng-Rao",
    ),
  ),
  affiliations: (
    (
      id: "1",
      name: "Politecnico di Milano",
    ),
  ),
  // cover-image: "./figures/polimi_logo.png",
  background-color: "#DDEEDD",
)

#include "chapters/chapter1.typ"
#include "chapters/chapter2.typ"
#include "chapters/chapter3.typ"
#include "chapters/chapter4.typ"
#include "chapters/chapter5.typ"
#include "chapters/chapter6.typ"
#include "chapters/chapter7.typ"
#include "chapters/chapter8.typ"
#include "chapters/chapter9.typ"
#include "chapters/chapter10.typ"
#include "chapters/chapter11.typ"
#include "chapters/appendix.typ"
