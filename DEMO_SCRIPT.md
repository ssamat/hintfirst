# Milestone 1 recording script (target 6–7 minutes)

## Before you record
- [ ] Gemini works locally: `GEMINI_MODEL` in secrets matches a model listed in Google AI Studio, and a question gets a live reply (admin view shows 🟢 live).
- [ ] Ran `python eval_gold_set.py --generate --write-cache` so the demo fallbacks match the real chapter.
- [ ] App deployed; link checked in an incognito window.
- [ ] Two browser tabs: the learner view (normal link) and the admin view (`?admin=1`, "Show safeguard details" on).
- [ ] Demo questions copied into a notes file.

---

## 1. Problem (0:00–0:40)
**[SCREEN]** The learner view, start page.

> Hi, I'm Shay. My project is Hint First, Then Explain: a tutor for adults working through an open textbook on their own, here Psychology 2e, Chapter 6 on learning. A general chatbot would give a confused learner a complete answer that isn't tied to their textbook and does the thinking for them. In Module 2, Bastani and colleagues found that unguarded AI raised practice scores but lowered later exam scores. My tool answers only from the reading, and it makes the learner think first.

## 2. Feedback and changes (0:40–1:40)
**[SCREEN]** One slide: "Feedback → What I changed."

> Here's what reviewers said and what I changed:
> - **Explain the problem more:** that's the example I just gave.
> - **Narrow the audience:** adult novices working through one unit, Psychology 2e Chapter 6.
> - **Kirschner, Sweller, and Clark argue for strong guidance:** agreed. So instead of a free switch to "explain," the tutor now gives graduated guidance: a hint, a bigger hint, then a full explanation.
> - **How do you keep hints accurate on unpredictable questions?** I'll show the safeguards in the demo.
>
> Other changes: the reading and the tutor now sit side by side, one section at a time, and learners can try an answer and get feedback before seeing the explanation.

## 3. Live demo (1:40–4:50)
**[SCREEN]** Learner view. Click **Start** (point out the consent checkbox is opt-in). Go to **6.3 Operant Conditioning**.

1. Ask: *What is the difference between negative reinforcement and punishment?*
   > I get a guiding question and a paragraph to reread, not the answer. Show sources takes me straight to that paragraph in the reading.
2. Click **Give me a bigger hint.**
   > If I'm still stuck, the next hint is more specific, but it still doesn't give the answer away.
3. Type an attempt, for example *"Negative reinforcement takes something away so the behavior goes up; punishment makes it go down,"* and submit.
   > Now I have to produce something. The tutor tells me what's on track and one thing to reconsider.
4. Click **Show the explanation**, then **Check yourself**, set confidence, and answer.
   > The explanation cites each paragraph. The quick check asks for my confidence before telling me if I'm right, which shows whether I actually know it.
5. **[SCREEN]** Switch to the admin tab. Ask *What is the capital of France?*, then *What part of the brain stores long-term memories?*
   > Off-topic questions are declined before the AI is even called. The safeguard details show why. A leak check also throws out any hint that copies the textbook word for word.

## 4. Evidence (4:50–5:20)
**[SCREEN]** Terminal output of `eval_gold_set.py`.

> I wrote 30 test questions, 10 of them tricky. The tutor did the right thing on 30 out of 30 and found the right section for all 24 answerable questions. The decline margin is small, so the AI's own "not in the reading" check is a second layer. Next, two reviewers will score each reply for correctness and for whether hints give the answer away.

## 5. Rationale (5:20–6:10)

> Why it's built this way:
> - **The hint ladder:** Module 2's help-seeking problem, where novices grab help too early, and Kestin and colleagues' finding that a tightly constrained, step-by-step tutor works.
> - **Trying an answer:** Module 4 says the best learning apps lean toward production over recognition, and Sadler in Module 5 says learners have to act to close the gap.
> - **The quick check and progress panel:** the testing effect from Module 5, and Module 1's point that tools should make thinking visible.
> - **One section beside the tutor:** Sweller's cognitive load theory.

## 6. Feedback I'd like (6:10–6:40)

> Your feedback would help most on three things:
> 1. Is the hint ladder the right length, or should the explanation unlock sooner?
> 2. Is the feedback on attempts specific enough to act on?
> 3. For Milestone 2, what would make the reflection at the end of each section worth doing?
>
> Thanks for watching!
