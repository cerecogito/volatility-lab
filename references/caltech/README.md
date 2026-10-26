# Caltech course notes (not included)

The author's copy of the *Pricing Options with Mathematical Models* slides (`All slides.pdf`,
285 pages) is kept **outside** this repository (in the parent folder), because it is copyrighted
course material. This folder is git-ignored except for this README.

How the notes were used:

- The section structure (§1 Overview to §33 Credit risk) guided which methods to build. See
  `../course_concept_map.md`.
- Three worked numerical examples from the slides are used as **independent test values**, i.e.
  numbers computed by the course, not by this code:
  - the single-period binomial call, 0.746 (§11);
  - the two-step European call 10.8703 and American put 2.4844 on the tree $u=1.1$, $d=0.9$,
    $r=5\%$ (§13);
  - the 20-day Microsoft replication table (§27). Here sigma = 0.317139, K = 55, r = 0, the
    loss with hedging is 0.1569 and the loss without is 11.5410. The spreadsheet uses the
    rounded T = 0.08 years. This was discovered while writing `tests/test_hedging.py`.
- Many slides with equations are images, not text. These were viewed directly for the BSM
  formula (§17), the B-S-M delta and replication example (§27) and the stochastic-volatility /
  Heston pages (§23).

To re-create the course-based tests no notes are needed: the numbers are in the tests.
