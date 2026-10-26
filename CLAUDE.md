# Fraud service — rules for Claude Code

Project: clean a dirty card-transaction CSV, train a Random Forest (with SMOTE), LightGBM and an LSTM, serve the
model with FastAPI (/predict with validation, one error shape, rate limiting), and wrap it in a Strands agent that
remembers the conversation across stateless /chat calls.

Layout: `data/` the CSV · `app/` the code (data.py, train.py, model.py, api.py, agent.py) · `tests/` pytest.

Commands: `pip install -r requirements.txt` · `python -m pytest` · `python -m app.train` · `uvicorn app.api:app --reload`

## How we work
1. **Code goes into the project files, never only into the terminal.** Every piece of work, including data
   exploration, is written as a function in the module where it belongs (exploration and cleaning in
   `app/data.py`, training in `app/train.py`, and so on). No `python -c` one-liners, no heredocs, no temporary
   scripts, no analysis I can't open in my editor.
2. **One plan, then do all of it.** For a task with several steps, show the plan once (short, numbered). When I
   approve it, implement every approved step in one go, including running the commands in the plan (tests,
   scripts), without stopping to ask between steps. Stop only if something fails or a decision comes up that
   the plan doesn't cover.
3. **Only what I asked for.** If I approve only some steps, do only those. Don't bring back steps or ideas I
   skipped, and don't suggest next steps or extra features, unless I raise that topic myself. Example: if I ask
   for an `is_refund` column, add the column, nothing else (no feature list, no refactor).
4. **Short reports.** When done, say in a few lines which files changed, what ran and the result. Explain the code
   only when I ask "explain".
5. **Always print, so I can check the values.** Every step prints what it did: row counts before and after,
   how many values it changed, `value_counts()` of a new or fixed column, min/max of numbers, shapes of the
   splits, and the metrics. When you run something, show me the printed output in your report.

## Engineering rules
6. Keep it simple: plain functions (a class only when it really helps), short files, the project's existing style.
7. No leakage: split by time first; SMOTE, imputers, encoders and scalers are fitted on the training data only.
8. PR-AUC, precision and recall, never accuracy alone; the threshold and the served model are chosen on
   validation, never on test. Iterative models (LightGBM, LSTM) use early stopping on validation.
9. Never change a test to make it pass. Fix the code, or tell me why the test is wrong.
