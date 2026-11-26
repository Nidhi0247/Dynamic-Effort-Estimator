# Practitioner study: do the fuzzy-rule explanations help people judge an estimate?

Takes about 20 minutes per participant. Run it with 8–12 people, ideally more.

## Who can take part
Developers, testers, scrum masters or final-year CS students who have used Jira or estimated agile tasks.
Record for each person: role, years of experience, and whether they have estimated with story points.
Check with your guide whether your institution needs ethics approval or a consent form.
Even when it doesn't, read out the consent text below and keep the responses anonymous.

> Consent text: "This is a 20-minute study about software effort estimates. You will see 12 real, anonymised tasks
> from open-source projects with a computer-generated estimate. There are no right or wrong opinions; we are testing
> the tool, not you. You can stop at any time. Responses are anonymous and used only for research."

## Design (within-subject, counter-balanced)
- 12 items in [stimuli.md](stimuli.md). Each is a real held-out requirement with the model's estimate and 95% range.
- Every participant sees 6 items **with** the "Why" rules and 6 **without**.
- **Group A:** odd items with the explanation, even items without. **Group B:** the reverse.
- Alternate participants between A and B.
- Shuffle the item order per participant if your form tool allows it.

## For each item, ask
1. **Decision:** "Would you accept this estimate, or adjust it?" (options: *much lower / a bit lower / accept /
   a bit higher / much higher*)
2. **Trust:** "How much do you trust this estimate?" (1 = not at all … 5 = completely)
3. *Explanation items only:*
   - **Understandable:** "The explanation is easy to understand." (1–5)
   - **Plausible:** "The explanation matches how I would reason about this task." (1–5)

## At the end (once)
- "What, if anything, would make the explanations more useful?" (free text)

## What I will compute (send me the responses as CSV)
- **Adjustment accuracy:** does the direction of the decision match where the actual effort fell relative to the
  estimate (from `answer_key_DO_NOT_SHOW.csv`)? Compared with vs without explanation, using a paired Wilcoxon test
  per participant.
- **Calibrated trust:** is trust higher for estimates that turned out accurate? Compared with vs without explanation.
- **Ratings:** understandability and plausibility (median, IQR), plus themes from the free text.

## Practical notes
- Google Forms works: one section per item, with the explanation pasted in the item's text for the "with" condition.
- **Never show the answer key** to participants.
- Do not explain the model before the study. Afterwards you may.
