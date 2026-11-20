# Milestone 2 recording script (target 6–7 minutes)

## Before you record
- [ ] Gemini works locally: a question gets a live reply (admin view shows "live"). The rewording redirect and the linked answers need the live tutor.
- [ ] `pytest -q` passes, and `python eval_gold_set.py --generate` has run, so the terminal shows the numbers below.
- [ ] Two browser tabs: the learner view (normal link) and the admin view (`?admin=1`, "Show safeguard details" on). Click **Clear session** in admin before you start.
- [ ] One slide ready: "Feedback → What I changed" (the table in section 2).
- [ ] Demo inputs copied into a notes file:
  - `What is negative reinforcement?`
  - `What is the difference between negative reinforcement and punishment?`
  - `what is negative reinforcement again?`
  - `Taking away something nice to stop a behavior is called what?`
  - `Out of food, water, sex, and money, which one isn't a primary reinforcer?`

---

## 1. Where the project is (0:00–0:30)
**[SCREEN]** Learner view, start page.

> Hi, I'm Shay. This is Hint First, Then Explain, a tutor for adults working through Psychology 2e, Chapter 6, on their own. The chat gives short, cited answers from the reading. Practice questions make the learner try first, with hints after. For Milestone 2 I worked through every piece of peer feedback, and I'll show each change in the app.

## 2. Feedback and changes (0:30–1:30)
**[SCREEN]** The slide.

| What reviewers said | What I changed |
|---|---|
| **Isak:** show the calibration feedback right after practice, not further up the page | A summary card appears right under the last question, with a pop-up notice |
| **Begum:** frame calibration supportively and show what to revisit, plus where learners were *less* confident than they needed to be | The card lists questions worth another look and "you knew more than you felt" |
| **Begum:** make the first hint more targeted by naming the key distinction | Hint 1 now names the distinction ("added or taken away?"); hint 2 says which sentence to look at |
| **Begum:** rewording a practice question can get around the paste check | Two new layers: naming 3+ of the options, and a Gemini "same question, reworded?" check |
| **Begum:** questions I wrote myself make the test weaker | Test files now record who wrote each question; classmates are writing items for the next round |
| **Eliseo:** show how answers build on each other and how memory carries over | Repeats bring back the earlier answer, new answers link to earlier questions, and new sections open with a recap |
| **Eliseo:** let learners save a chat response as highlighted text | **Save to notes** pins an answer under the paragraph it cites |

> Reviewers also agreed with two choices I'm keeping: direct answers in the chat with hints only in practice, and one section at a time.

## 3. Live demo (1:30–5:00)

### 0. A quick tour of the screen
**[SCREEN]** Point at the layout.
> The screen stays simple: the reading on the left, the tutor on the right. Each paragraph has a 💬 button to ask about it. Extras like the Sources box and quick reviews live under ⚙️ Settings, along with a short "How this works".

### a. Save to notes (Eliseo)
**[SCREEN]** Learner view. Click **Start**, then **Next →** twice to reach **6.3 Operant Conditioning**.

1. Ask: *What is negative reinforcement?*
   > A short answer, each sentence cited to a paragraph.
2. Click **📌 Save to notes** under the answer. Then click a citation like [6.3 ¶4] in the answer.
   > The paragraph it cites turns blue in the reading, and my note sits right under it. Every citation is a link: one click highlights the paragraph and scrolls to it. The 📌 button at the top keeps every note I've saved.

### b. Memory (Eliseo)
3. Ask: *What is the difference between negative reinforcement and punishment?*
   > The tutor knows what I asked earlier, so the answer can start from it instead of starting over.

   (If the answer links back, point to that phrase. If it doesn't, say: "The earlier question goes into the prompt, so it can link back when that helps.")
4. Ask: *what is negative reinforcement again?*
   > I already asked this, so instead of a new answer it brings back the earlier one and says where I asked it. That way I don't end up asking the same thing over and over.
5. Click **🔄 Explain it differently**.
   > If the first answer didn't land, I can ask for a new angle: simpler words and an everyday example.

### c. Rewordings can't get around practice (Begum)
6. Click **Mark section done**. The panel switches to Practice. Switch back to **Ask the reading**.
7. Paste: *Taking away something nice to stop a behavior is called what?*
   > That's practice question 2, reworded. A copy check alone missed it, so the tutor now asks Gemini whether this is the same question, and it sends me back to practice.
8. Paste: *Out of food, water, sex, and money, which one isn't a primary reinforcer?*
   > This one lists the answer options, and that's caught even offline.
9. **[SCREEN]** Switch to the admin tab, go to 6.3, click **Mark section done**, then ask the same two questions in **Ask the reading**. Point at the safeguard line ("Gemini: same question reworded" / "names 4 of its options").
   > A general question like "What is shaping?" still gets a normal answer: it's about the topic, not the practice question.

### d. A sharper first hint (Begum)
10. **[SCREEN]** Learner tab, **Practice**. On *"________ is when you take away a pleasant stimulus to stop a behavior"*, click **I'm stuck**, then **Hint**.
   > The first hint now names the key distinction: is something being added or taken away, and is the goal to increase or decrease the behavior? Then it points to the paragraph.
11. Click **Bigger hint**.
    > The bigger hint tells me which sentence to look at, without saying what it says.
12. Click **Worked answer**, then type one sentence in **Explain it back**, e.g. *"Something pleasant gets taken away, and the goal is to make the behavior stop."* Click **Check my reason**.
    > After the worked answer, I have to say in my own words why the answer is right. If I had picked a wrong option first, it would ask why that one is wrong. That turns a worked example into active processing.

### e. The summary card (Isak and Begum)
13. Answer all five questions. Use this pattern so every part of the card shows up:

    | Question | Answer | Confidence |
    |---|---|---|
    | Two forms of associative learning | right (classical; operant) | **2** |
    | Take away a pleasant stimulus | right (negative punishment) | 4 |
    | Not a primary reinforcer | **wrong** (pick "food") | **5** |
    | Successive approximations | right (shaping) | 4 |
    | Slot machines | right (variable ratio) | 4 |

    > As soon as I answer the last one, a notice pops up and the summary appears right here, so I don't have to scroll for it.
14. Read the card.
    > It shows how sure I felt and how many I got right. Under "Worth another look" it lists the question I missed, with a button to its paragraph. "You knew more than you felt" tells me I got question 1 right while rating it 2. It's meant to read like guidance, not a grade.

### f. A new section remembers the last one (Eliseo), and spaced review
15. Click **Next →**, skip the reflection, click **Continue**.
    > In 6.4 the chat opens with a short recap: what I asked in 6.3 and how many notes I saved. And a quick review card brings back the question I missed in 6.3, with the options reshuffled. Coming back to it later in the session is spaced retrieval, which sticks better than rereading right away.

## 4. Evidence (5:00–5:40)
**[SCREEN]** Terminal output of `python eval_gold_set.py --generate`.

> Same checks as before, plus the new ones:
> - Chat behavior is still 30 of 30 offline. All 32 close copies of practice questions get redirected.
> - On 16 looser rewordings, the copy check alone caught 2. With the options check and the Gemini check, it's 14 of 16. The two misses barely share any words with the original question, so they never reach Gemini.
> - One ordinary question now gets redirected: "give me the definition of extinction." The definition of extinction is that practice question's answer, so I'd argue that's the right call. Two same-idea questions, "What are the four steps of modeling?" and "What did the Bobo doll study find?", also go to practice while those items are open. Once they're finished, the chat answers them.
> - All 16 first hints pass the leak check and name the key distinction. Bigger hints pass 15 of 16. My first wording for the bigger hint made Gemini quote the sentence, so I tightened it.
> - Begum pointed out that I wrote all the test questions myself. Each test question now records who wrote it, and classmates are writing tricky questions and rewordings, so the next report shows their questions separately from mine.

## 5. Rationale (5:40–6:20)

> Why each change fits the course:
> - **The summary card:** self-regulated learning, Module 1. Learners need to see where how sure they felt didn't match how they did, in both directions. Sadler, Module 5: feedback has to show *how* to close the gap, so the card links each question to its paragraph.
> - **The first hint:** productive help-seeking, Module 2. A targeted first hint makes the next step clear without taking it for the learner.
> - **Notes beside the reading:** less split attention, Module 0. The note sits next to the paragraph it came from.
> - **Chat memory:** answers build on what the learner already asked, so the help adds up over the session.
> - **The rewording check:** protects the testing effect. Retrieval only helps if the learner does the retrieving.

## 6. What's next and feedback I'd like (6:20–7:00)

> Next, I'll run a small pilot with three to five learners on one section, using the classmate-written questions as the test set.
>
> Feedback that would help most:
> 1. Does the summary card read as guidance? Is "You knew more than you felt" useful, or just noise?
> 2. Should a repeat question bring back the old answer, or explain it a new way?
> 3. Is redirecting "give me the definition of extinction" to practice the right call?
>
> Thanks for watching!
