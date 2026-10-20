POLICY_GEN_INSTRUCTIONS = """\
You are part of an intelligent team of LLMs collaborating on a binary
classification task.

Given:
- A task description.

Task:
- Write a prompt/instructions template for another LLM (the policy generator).
- The policy generator receives the task description and a small batch of
  labelled samples (some YES, some NO). It sees nothing else.
- It must write policies: general guidance that a decision-maker could use to
  decide whether the answer to the task is YES for any new sample.

The template must tell the policy generator that policies should be:
  1. Guidance, not facts: say what to weigh or look for (for example "Give
     weight to ..."), never a claim that is simply true or false of one sample.
  2. Generalizable: capture broader patterns, not details of a single sample.
  3. Transferable: focus on signals that apply beyond the samples shown.
  4. Concise: one clear, actionable sentence each.
  5. Non-overfitting: never mention a specific sample, name or value.
- The generator is free to write as many policies as the batch supports. A
  later step merges and trims them, so it must not worry about a limit.
- Policies may point towards YES or towards NO.

Return:
- Only the text of the prompt/instructions template for the policy generator.
"""

POLICY_MERGE_INSTRUCTIONS = """\
You are part of an intelligent team of LLMs collaborating on a binary
classification task.

Given:
- A task description.
- A list of policies. Each is guidance for deciding whether the answer to the
  task is YES. They were written independently from different batches of
  samples, so many overlap or repeat one another.

Task:
- Merge the policies into one list of AT MOST {max_policies} policies.
- Combine policies that say the same thing or nearly the same thing.
- Keep the most general and most useful guidance; drop policies that are
  vague, redundant, or specific to a single case.
- Keep each policy a concise, actionable one-sentence piece of guidance.
- Do not invent guidance that none of the input policies support.

Return the final list of policies.
"""

POLICY_JEV_STATE = """\
Task: {task_description}

Sample:
{sample}"""

POLICY_JEV_QUESTION = """\
Use this policy as guidance for your decision: {policy}
Is the answer to the task YES?"""
