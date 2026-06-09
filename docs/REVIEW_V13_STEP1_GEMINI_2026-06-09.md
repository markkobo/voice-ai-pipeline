1. **One-line verdict**
Hold — this doesn't fully implement Step 1 of the plan, blocking Step 2, and has a latent bug with API key handling that will cause silent mass failure.

2. **Real bugs**
-   **`scripts/v13_build_dataset.py:213`**: The script initializes the OpenAI client but only uses it deep inside a loop. If the `OPENAI_API_KEY` is missing from `.env`, `OpenAI()` doesn't fail, but every subsequent call to `client.audio.transcriptions.create()` will fail. The `try/except` in `transcribe_one` will catch this, log an error, and continue, resulting in a long, expensive-feeling run that produces no output and no clear root cause diagnosis. The script should fail fast at startup if the API key is not configured.

3. **Pipeline-correctness issues**
-   **`scripts/v13_build_dataset.py`**: The script omits Step 1d from the V13 plan. The plan requires a `v13_prepare.py` equivalent to tokenize the audio and produce a `train_prepared.jsonl` file. This commit only produces `train.jsonl`, which is an intermediate artifact. The training code in Step 2 requires the prepared data, so this commit leaves the pipeline in a broken, intermediate state.

4. **Latent issues**
-   **`scripts/v13_build_dataset.py:146`**: The transcription loop has no retry logic. For a persona with significant audio, the script is likely to fail on many chunks due to OpenAI API rate limiting or transient network errors. A simple backoff-and-retry wrapper (e.g., using `tenacity`) around `transcribe_one` would make the script dramatically more robust.
-   **`scripts/v13_build_dataset.py:243`**: The progress-saving mechanism (`_write_jsonl_atomic`) re-reads, re-sorts, and re-writes the entire dataset every 10 new records. While robust, this is inefficient (`O(N log N)`) and will become noticeably slow as the dataset grows to thousands of chunks. Appending to a file and sorting once at the end would be more performant.
-   **`scripts/v13_build_dataset.py:130-133`**: Long audio chunks are trimmed to `MAX_CHUNK_MS`. This is better than dropping them, but might create unnatural sentence cut-offs that affect transcription quality and downstream TTS prosody. A warning is logged, but it might be better to split the long chunk again using a more aggressive silence threshold.

5. **Tests not yet written**
-   The file-finding logic in `find_persona_recordings` is untested. It relies on a string-matching heuristic (`f"_{persona}_" in d.name`) that could fail if persona names are substrings of one another (e.g., `test` vs `supertest`). A unit test with a mocked directory structure would validate this critical input stage.
-   The idempotency logic (skipping existing chunks/transcripts) is not explicitly tested. A test that runs the script twice on the same input and asserts the second run is a no-op would confirm correctness.

6. **Best 1-2 improvement suggestions**
1.  **Fail-fast on missing API key:** Before the main loop, add a check for the `OPENAI_API_KEY` environment variable. If it's missing or empty, print a clear error message and `sys.exit(1)`. This turns a confusing silent failure into an immediate, actionable error.
2.  **Add retry logic for API calls:** Wrap the `client.audio.transcriptions.create` call with an exponential backoff retry decorator (e.g., from the `tenacity` library). This will handle transient API errors and rate limiting, which are virtually guaranteed on any non-trivial dataset size.
