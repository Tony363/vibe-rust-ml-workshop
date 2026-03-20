# Speaker Notes

## Session: Vibe Coding a Rust ML Pipeline (45 min)

Target audience: AI/vibe coding enthusiasts. Do NOT assume Rust expertise. Keep language accessible and jargon-free. When Rust-specific concepts come up, explain them in plain terms.

---

### 0:00-0:08 -- Hook + Context

**Open with a question to the room:**

> "Who here has written Rust before? Python? Used an AI coding assistant?"

Use the show of hands to calibrate. If most people are Python users, lean into comparisons. If some have used AI assistants, ask what they liked or found frustrating.

**Talking Points:**

- **What is "vibe coding"?**
  - Coding guided by AI assistants like Claude Code, where you let the AI handle the syntax while you focus on intent.
  - You describe what you want in plain language. The AI writes the code. You review, iterate, and steer.
  - It flips the traditional model: instead of memorizing APIs, you have a conversation with your tools.
  - This is not "no code" -- you still need to understand what the code does. You just don't have to write every character yourself.

- **Why Rust for ML?**
  - Memory safety without a garbage collector. Your ML pipeline won't silently corrupt data or segfault at 3 AM.
  - Performance that rivals C/C++. When you need to process millions of records, Rust delivers.
  - The compiler is like a built-in AI safety net -- it catches entire classes of bugs (null pointers, data races, use-after-free) before your code ever runs.
  - For the Python folks: imagine if Python's type hints were mandatory AND the interpreter refused to run code with potential bugs. That is Rust's compiler.

- **The Rust ML ecosystem:**
  - **Linfa** -- the scikit-learn equivalent for Rust. Provides classic ML algorithms: decision trees, clustering, linear regression, SVMs, and more.
  - **Burn** -- a deep learning framework built natively in Rust. Think PyTorch but with Rust's safety guarantees.
  - **Candle** -- Hugging Face's Rust ML framework, focused on inference and lightweight model serving.
  - The ecosystem is younger than Python's, but it is growing fast and is already production-ready for many use cases.

- **What we will build today:**
  - An Iris flower classifier -- the "hello world" of ML.
  - We will load data, train a decision tree, evaluate it, and produce beautiful terminal output.
  - All in about 130 lines of Rust, written with AI assistance.

**Transition:** "Let's stop talking about it and start building. I'm going to open my terminal and we will scaffold this project together."

---

### 0:08-0:22 -- Live Build: Scaffold to Data to Training (Acts 1-2)

**Act 1: Scaffold the project (about 3 minutes)**

- Run `cargo new vibe-rust-ml-workshop` and show what gets generated.
- Explain briefly: `Cargo.toml` is like `package.json` or `requirements.txt` -- it declares your dependencies. `src/main.rs` is your entry point.
- Walk through adding dependencies to `Cargo.toml`:
  - `linfa` and `linfa-trees` for ML
  - `linfa-datasets` for the Iris dataset
  - `comfy-table` for pretty terminal output
- Run `cargo build` and let the audience see dependencies download. Use this compile time to talk:
  - "Rust compiles ahead of time. This initial build is slow because it is compiling every dependency from source. Subsequent builds are much faster."
  - "This is also a good moment for vibe coding -- while it compiles, you can plan your next move."

**Git tag: step-1-scaffold**

**Act 2: Load data and train the model (about 11 minutes)**

- Load the Iris dataset using `linfa_datasets::iris()`.
- Show how the compiler catches errors immediately if you use the wrong type or forget an import.
  - Key message: "Watch how the compiler guides us -- it is like pair programming with a very strict partner."
  - If an error appears, do NOT rush to fix it. Read the error message aloud. Rust's compiler errors are famously helpful. Let the audience see that the compiler tells you exactly what is wrong and often suggests the fix.
- Add a train/test split. Explain briefly:
  - "We split the data so we can test our model on data it has never seen. This is ML 101 -- you do the same thing in scikit-learn."
- Build and train a decision tree classifier.
  - Walk through the builder pattern: `DecisionTree::params().max_depth(Some(4)).fit(&train)`.
  - Explain: "This reads almost like English. Set max depth to 4, then fit on our training data. The builder pattern is common in Rust."
- Run the code and show it compiles and trains.

**Git tag: step-2-data** (after data loading works)
**Git tag: step-3-training** (after model trains successfully)

**If things go wrong:**

Recovery commands:
```
git stash && git checkout step-2-data
git stash && git checkout step-3-training
```

Say to the audience: "One of the beautiful things about live coding is that things break. But we have checkpoints. Let me jump ahead so we do not lose momentum."

**Transition:** "We have a trained model. But a model that does not tell you how well it works is not very useful. Let's add evaluation."

---

### 0:22-0:33 -- Live Build: Evaluation + Pretty Output (Act 3)

**Talking points and steps:**

- **Add predictions:**
  - Use the trained model to predict on the test set.
  - Compare predictions to actual labels.

- **Calculate accuracy:**
  - Walk through the logic: count how many predictions match the true labels, divide by total.
  - Show the accuracy printed to the terminal.

- **Build a confusion matrix:**
  - Explain what a confusion matrix is for anyone unfamiliar: "It shows you not just how often the model was right, but how it was wrong. Did it confuse setosa with versicolor? This is more informative than a single accuracy number."
  - Build the matrix manually -- it is just a 3x3 grid of counts.

- **Add comfy-table formatting:**
  - Use `comfy-table` to render the confusion matrix as a nicely formatted ASCII table in the terminal.
  - This is a great visual moment -- the audience sees a professional-looking output appear.
  - Key message: "This is production-quality output from about 130 lines of Rust. No Jupyter notebook, no matplotlib. Just a binary you can ship anywhere."

- **Optional: add per-class metrics** (precision, recall) if time allows.

**Git tag: step-4-complete**

**If things go wrong:**

Recovery command:
```
git stash && git checkout step-4-complete
```

**Transition:** "Now here is where it gets fun. I want YOU to tell me what to build next."

---

### 0:33-0:40 -- Audience Challenge

**Setup:**

Say to the audience: "We have a working ML pipeline. Now let's do some real vibe coding. What should we try next? Shout it out."

**Suggested options to offer if the room is quiet:**

- Change `max_depth` from 4 to 3 or 10 -- how does accuracy change?
- Switch from Gini impurity to Entropy as the split criterion -- does it matter?
- Add a feature importance display -- which measurements matter most for classifying irises?
- Change the train/test ratio from 80/20 to 60/40 -- how robust is our model?
- Add color to the terminal output using ANSI codes or the `colored` crate.

**How to run this segment:**

1. Take the audience suggestion.
2. Describe the intent to Claude Code in plain language, out loud, so the audience hears the "vibe" part of vibe coding.
3. Let the AI generate the code change.
4. Review it together with the audience. Ask: "Does this look right? What do you notice?"
5. Run it. Celebrate or debug together.

**Tips:**

- If the suggestion is too complex for the remaining time, say so honestly and simplify it. "Great idea -- let's do a simpler version of that."
- If the AI makes a mistake, that is a teaching moment. Show how you correct course.
- Keep energy high. This is the interactive highlight of the session.

---

### 0:40-0:45 -- Wrap-up + Q&A

**Talking Points:**

- **Honest limitations of the Rust ML ecosystem:**
  - It is younger than Python's. Linfa covers the basics (decision trees, clustering, linear models) but it is not scikit-learn's full breadth.
  - Deep learning in Rust is possible (Burn, Candle) but the ecosystem of pre-trained models, tutorials, and community support is still growing.
  - If you need cutting-edge model architectures or massive pre-trained model zoos, Python is still the pragmatic choice today.

- **The hybrid pattern -- the practical takeaway:**
  - Prototype in Python (fast iteration, huge ecosystem, notebooks).
  - Optimize hot paths in Rust (data preprocessing, feature engineering, serving).
  - This is not an either/or choice. Use both.

- **"Systems around models" -- the real opportunity:**
  - Most production ML is not the model itself. It is the data pipeline, the feature store, the serving layer, the monitoring.
  - Rust excels at these "systems around models" -- high-performance data ingestion, reliable serving with predictable latency, safe concurrent processing.
  - You do not need to rewrite your neural net in Rust. Use Rust for the infrastructure that makes your model production-ready.

- **Resources to share:**
  - Linfa documentation: https://docs.rs/linfa
  - Burn (deep learning in Rust): https://burn.dev
  - Candle (Hugging Face Rust ML): https://github.com/huggingface/candle
  - This workshop repo with all checkpoint tags for self-paced learning.

**Close with:**

> "The combination of vibe coding and Rust gives you something new: you get the AI to handle the syntax, the compiler to handle correctness, and you get to focus on what matters -- solving the actual problem. Thanks for building with me today."

**Open for Q&A.** If no questions, prompt with:

- "What surprised you most about Rust's compiler today?"
- "Who is going to try vibe coding their next project?"

---

### Recovery Instructions

If you fall behind or something breaks during the live demo, use these checkpoints to jump to any stage of the project. Each tag represents a working state.

```
git stash                        # save any in-progress work
git checkout step-1-scaffold     # jump to scaffold
git checkout step-2-data         # jump to data loading
git checkout step-3-training     # jump to training
git checkout step-4-complete     # jump to final version
git checkout master              # return to main
```

**Before the workshop:**

- Verify all tags exist: `git tag -l`
- Test each checkpoint compiles: check out each tag and run `cargo build`
- Pre-download all dependencies: run `cargo build` at least once so crates are cached. A live `cargo build` with no cache on conference Wi-Fi is risky.
- Have the final version (`step-4-complete`) ready to screen-share as a backup if everything goes sideways.

**During the workshop:**

- If a recovery jump is needed, be transparent: "Let me jump to our checkpoint so we can keep moving." The audience respects honesty over pretending nothing went wrong.
- After jumping, briefly re-orient: "OK, we are now at the stage where data is loaded. Let me show you what this code does before we move on."

---

### Key Messages to Reinforce Throughout

1. **The Rust compiler is your AI safety net.** It catches entire classes of bugs at compile time -- null pointer dereferences, data races, use-after-free, type mismatches. These are bugs that would be silent runtime errors in other languages.

2. **Vibe coding + Rust = the AI handles syntax, the compiler handles correctness.** You focus on intent and architecture. The AI writes the code. The compiler verifies it. This is a powerful workflow.

3. **The hybrid Python+Rust pattern is practical today for production ML systems.** You do not have to choose one language. Use Python where it shines (experimentation, model training, notebooks) and Rust where it shines (performance, reliability, deployment).

4. **"Systems around models" -- you do not need to rewrite your neural net in Rust.** The biggest opportunity for Rust in ML is not replacing PyTorch. It is building the reliable, high-performance infrastructure that surrounds your models: data pipelines, feature engineering, serving layers, monitoring systems.
