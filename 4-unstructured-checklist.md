# ❤️ Structured Data Extraction from Unstructured Documents

### Introduction

In the last two lessons, you mastered **structured outputs** with Pydantic and set up basic **evaluations** for an LLM-driven SQL generator. Now we’ll **build on those concepts** in a new context: extracting structured information from free-form text documents. Think of all the unstructured text your enterprise deals with – emails, reports, **resumes** – and how much manual effort goes into pulling out key facts. If we can teach an LLM to read a raw resume and produce a neatly structured summary (name, skills, experience, etc.), we save time and reduce errors for our HR teams.

In this lesson, we’ll focus on resumes as our example unstructured documents. You’ll learn how to define a **checklist of fields** (via a Pydantic model) that the LLM must fill in, just like a form. We’ll also explore how to **route prompts** to the correct schema when dealing with different document types (so the model knows which checklist to use), how to handle **missing information** gracefully, and how to **evaluate** the quality of the extracted data with simple heuristics. By the end, you’ll be able to turn a blob of resume text into a structured JSON object with all the important details – automatically.

---

### Resumes: From Unstructured Text to Key-Value Fields

A resume is essentially a free-form narrative of a person’s professional history. One candidate might start with a personal summary, another with work experience or education – there’s no fixed format. For an LLM, however, we want to **impose structure** on this chaos. We need specific fields like **name**, **skills**, **years of experience**, **education level**, **last job role**, **salary expectations**, and **number of projects**. These fields act like a *checklist* for the model: no matter how the information is buried in the text, the model should extract and populate each field (or mark it as not available).

Why these fields? Imagine you’re a recruiter skimming a resume – this is the kind of information you’d jot down in a structured form or an applicant tracking system. By having the LLM output these items in a consistent format, downstream systems (databases, search indexes, matching algorithms) can readily consume the data. It’s the classic transformation of unstructured data to structured data, powered by prompt engineering and the model’s understanding of resumes.

Let’s consider a quick example. Suppose we have the following snippet from a resume:

*“Jane Doe is a seasoned software engineer with 5 years of experience in full-stack development. She holds a Master’s degree in Computer Science. Currently a Senior Developer at TechCorp, Jane has led 3 major projects and is seeking new opportunities with a salary expectation of around $120k. Her skills include Python, JavaScript, React, and AWS.”*

From this text, a human would extract things like:

- **Name:** Jane Doe
- **Skills:** [Python, JavaScript, React, AWS]
- **Years of Experience:** 5
- **Education Level:** Master’s in Computer Science
- **Last Job Role:** Senior Developer
- **Salary Expectations:** $120k
- **Projects Count:** 3

Our goal is to get the LLM to produce exactly that kind of structured data automatically. How? By **defining a schema** and asking the model to follow it.

---

### Designing a Pydantic Schema as a Checklist of Fields

In Lesson 2, you saw how Pydantic models can enforce a response format on the LLM (thanks to OpenAI’s Beta API). We’ll do the same here by defining a Pydantic model for resume information. This model’s fields are essentially our extraction **checklist**. Each field comes with a description to guide the model on what to put there.

For our resume extraction, we can define a class `ResumeInfo` with the fields we want:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ResumeInfo(BaseModel):
    name: str = Field(..., description="Full name of the candidate")
    skills: List[str] = Field(..., description="List of technical or professional skills mentioned")
    years_experience: int = Field(..., description="Total years of professional work experience")
    education_level: str = Field(..., description="Highest education attained, e.g. Bachelor's, Master's, PhD")
    last_job_role: str = Field(..., description="Title of the most recent job or current role")
    salary_expectations: Optional[int] = Field(None, description="Desired salary if stated (annual, in USD)")
    projects_count: Optional[int] = Field(None, description="Number of projects mentioned or led, if any")
```

A few things to note here:

- We use `Optional[int]` with a default of `None` for fields that might not appear in every resume (salary expectations and projects count). This tells the model (and the parser) that it’s okay for those to be missing – they can be null.
- Each field has a **description**. These descriptions are *invisible to the model in terms of tokens* (the OpenAI API doesn’t count them against your prompt length), but they serve as instructions for what content to put in each field. Essentially, the model gets a clear definition of each slot it needs to fill. For example, it knows `skills` should be a list of skills, `education_level` is looking for something like "Bachelor's" or "Master’s," and so on.
- The model output will be validated against this schema. If the model doesn’t follow it (say it leaves out a required field or puts text where an int is expected), the Pydantic parser will raise an error or fail to parse. This makes the output **reliable** – we either get a correctly structured object or we know it didn’t meet the requirements.

**How do we prompt the LLM with this schema?** With the OpenAI Python client, it’s straightforward: we pass `response_format=ResumeInfo` in the `chat.completions.parse` call. Here’s a pseudo-code example of using the model to parse resume text:

```python
import openai

# Imagine resume_text is a string containing the resume
resume_text = """Jane Doe is a seasoned software engineer with 5 years of experience ... (etc.)"""

messages = [
    {"role": "system", "content": "You are an expert HR assistant. Extract key information from the user's resume."},
    {"role": "user", "content": resume_text}
]

response = openai.ChatCompletion.create(
    model="gpt-4-0613",  # for example
    messages=messages,
    functions=[{"name": "extract_resume_info", "parameters": ResumeInfo.schema()}],  # Beta function call with schema
    function_call={"name": "extract_resume_info"}
)
parsed_output = ResumeInfo.parse_obj(response["choices"][0]["message"]["function_call"]["arguments"])
print(parsed_output.json(indent=2))
```

The above code would send the resume text to the model with an instruction to use the `ResumeInfo` schema (using OpenAI’s function calling style under the hood). The result we get back is directly parsed into a `ResumeInfo` object. We didn’t even have to say “return JSON” – by providing the schema, the model **knows** to output a JSON that matches it. This means no post-processing to split strings or regex out values; the model does the heavy lifting.

If we ran this on Jane’s snippet, the printed JSON might look like:

```json
{
  "name": "Jane Doe",
  "skills": ["Python", "JavaScript", "React", "AWS"],
  "years_experience": 5,
  "education_level": "Master’s in Computer Science",
  "last_job_role": "Senior Developer",
  "salary_expectations": 120000,
  "projects_count": 3
}
```

Exactly the structured summary we want! The model has effectively turned an unstructured paragraph into a structured record.

> [!TIP]  
> **Field Descriptions are Prompt Magic:** The `Field(..., description="...")` text acts like a mini-prompt for each field. For instance, even if the resume doesn’t literally say “Years of experience: 5,” the description “Total years of professional work experience” nudges the model to infer that from context (like seeing “5 years of experience” in the text). This is a great way to embed guidance without bloating your main prompt – and as a bonus, these descriptions don’t count toward the token limit.

---

### Routing Prompts for Different Document Types

In real-world applications, you might not just deal with resumes. Perhaps you have multiple types of documents: resumes, job descriptions, cover letters, client profiles, etc. Each type of document would have its own schema or checklist of fields to extract. We need the system to **automatically pick the right schema** based on the document content – this is where **routing prompts** come in.

A routing prompt is essentially a small precursor step (or a clever instruction) that guides the model on *which* extraction format to use. There are a couple of ways to implement routing:

1. **Two-Stage Approach (Classifier + Extractor):** First, use an LLM (or even a simple rule) to classify the document type. For example, you could prompt the model: *“The user will provide a document. Determine if it’s a resume, a job description, or an email.”* Once you get the type, you then choose the corresponding Pydantic model for extraction and run the second stage. For instance, if stage one says "resume," you use `ResumeInfo`; if it says "job_description," you use a different `JobDescriptionInfo` schema in the second call. This approach is straightforward and modular – you have one prompt for routing, another for extraction.

2. **Single-Stage Prompt with Conditional Instructions:** In some cases, you can merge the logic by giving the model a system message like: *“If the document is a resume, extract according to Schema A; if it’s a job posting, extract according to Schema B.”* and then listing both schemas. However, this can be tricky – it makes the prompt larger and risks confusion. Usually, the two-stage approach is easier to maintain and debug.

Imagine we have both `ResumeInfo` and `JobDescriptionInfo` Pydantic models defined. A pseudo-code for routing might look like:

```python
doc_text = get_document_text()  # This could be any text

# Simple heuristic: if certain keywords appear, decide type
if "education" in doc_text.lower() or "experience" in doc_text.lower():
    chosen_schema = ResumeInfo
elif "Responsibilities" in doc_text or "Responsibilities:" in doc_text:
    chosen_schema = JobDescriptionInfo
else:
    chosen_schema = GenericDocInfo  # a fallback general schema

result = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[{"role": "system", "content": "Extract relevant information from the document."},
              {"role": "user", "content": doc_text}],
    functions=[{"name": "extract_info", "parameters": chosen_schema.schema()}],
    function_call={"name": "extract_info"}
)
extracted = chosen_schema.parse_obj(result["choices"][0]["message"]["function_call"]["arguments"])
```

In this pseudo-code, we used a simple keyword check to decide which schema to use. In practice, you might do a first LLM call to get a more nuanced classification (especially if document types overlap in content). But the essence is: **route to the correct checklist.** We don’t want to force a resume schema on a document that isn’t a resume, or vice versa.

For our lesson focus, we’ll assume we’re dealing with resumes, so the routing is simple. But it’s good to know how you’d extend this idea. In a large enterprise setting, having a “document router” is very powerful – it’s like having an AI switchboard that ensures the right processing is applied to the right document without manual intervention.

> [!CAUTION]  
> **Be Clear on Document Type:** If you skip a routing step and just always assume, say, a resume format, the model might get confused when the input diverges from expectations. For example, feeding a job description into a resume extractor could yield nonsense or very incomplete data. Always either route explicitly (via code or a classification prompt) or include clear indicators in your system message about what type of document to expect. This will save you from bizarre outputs or hallucinated fields.

---

### Handling Missing Fields and Incomplete Info

Real resumes vary widely. Not every candidate will state their **salary expectations**, and some might not clearly list a **projects count**. How should our LLM extractor handle cases where a field from our schema isn’t explicitly present? This is an important part of designing the prompt and schema.

First, because we defined certain fields as `Optional[...] = None` in our Pydantic model, we have implicitly told the model “it’s okay if this is blank.” When using OpenAI’s structured output, the model will know it *can omit* optional fields or set them to null. If a required field (like name or skills) is truly missing from the text, the model might try to infer or it might leave it blank – but since it’s required in the schema, the safer behavior (to avoid a parsing error) is often to output something like an empty string or a placeholder. We can guide this behavior in our instructions.

It’s good practice to **mention in the system prompt how to handle missing data**. For example, we could say: *“If a field is not mentioned, you may output `null` or `"N/A"` for that field.”* This explicitly signals to the model that it shouldn’t hallucinate or make something up just to fill the slot. Instead, a null (or `"N/A"`) is the expected safe output for missing info. The choice between `null` vs `"N/A"` might depend on what your downstream system expects – `null` (JSON null) is nice because it’s clearly not a string of actual data. In our scenario, null is a good choice via the Pydantic schema (since those optional fields default to None).

For example, if a resume has no mention of salary expectations, we’d want:

```json
{
  "name": "John Doe",
  "skills": [...],
  "years_experience": 7,
  "education_level": "Bachelor's in Marketing",
  "last_job_role": "Marketing Manager",
  "salary_expectations": null,   <-- model explicitly says this is not provided
  "projects_count": 2
}
```

And if a resume doesn’t list any distinct projects, perhaps `projects_count` would be null as well.

It’s better to output **null** or a placeholder than to guess. An LLM might be tempted to infer salary from context (“senior engineer in SF, probably $120k+”), but that’s not in the resume – it would be inventing data. Our instructions and schema design should discourage that. The combination of making a field optional and telling the model how to handle missing cases usually does the trick.

Finally, keep an eye on how the model behaves with required fields that are missing. For instance, if the resume text truly never states the candidate’s name (maybe it’s cut off or in an image?), the model might still try to fill the `name` field. In such cases, it could default to something like an empty string or a guess (“Unknown”). There’s no perfect solution if your input is incomplete, but being aware of this helps – you might do a post-check, e.g., if the extracted name is an empty string or not a real name, flag that resume for manual review. In general, though, with resumes we expect these key fields to be present most of the time.

---

### Evaluating the Extraction Results

After setting up an LLM to extract resume fields, we need to **evaluate how well it’s doing**. This isn’t a simple right/wrong check like SQL query accuracy, because here we’re dealing with a variety of field types and potentially subjective matches. We’ll start simple, with a small set of manually curated examples, and measure field-by-field performance.

**1. Create a Ground-Truth Dataset:** Just as we did for SQL in the previous lesson, we need some examples with known correct answers. For extraction, this means taking a handful of resumes and manually writing down the expected structured data for each. Even ~10 examples is enough to surface obvious issues. You might have a CSV with two columns: `resume_text` and `expected_json` (where `expected_json` is the structured data you compiled by reading the resume yourself).

**2. Run the LLM Extractor on Each Example:** Use your pipeline to generate a `ResumeInfo` output for each resume in the test set. Now you have model outputs and your ground-truth answers for comparison.

**3. Compare Field by Field:** For each resume example and each field, check if the model’s answer matches the expected answer. This is where “soft matching” comes in:
- For **text fields** like `name` or `education_level`, an exact string match might be too strict due to minor differences (e.g., expected "Master of Science in Computer Science" vs model output "Master’s in Computer Science"). A **soft match** could mean checking that certain keywords or phrases are present. For instance, you could lowercase both and see if one contains the other, or use a similarity metric (Levenshtein distance or token overlap). If the model output is *semantically* the same as the truth, we count it as correct.
- For **list fields** like `skills`, you might ignore order and just check that the sets overlap significantly. If your expected skills are ["Python", "React", "AWS"] and the model gives ["Python", "AWS", "React"], that’s a perfect match (order doesn’t matter). If it gives ["Python", "AWS"], missing one, that’s partially correct. If it adds extra irrelevant skills, that’s an error. You can decide on the threshold – maybe you require an exact set match or allow one missing/extra within reason.
- For **numeric fields** like `years_experience` or `projects_count`, it’s usually clearer: 5 vs 5 is a match. But consider cases like a resume says "5+ years" and you expected 5 while the model returned 6 (maybe it interpreted “5+” as 6). Is that wrong? Arguably, yes, since the resume didn’t explicitly say 6. But if a resume says "3-4 years experience" and the model outputs 4, it chose the upper bound. You might still mark that as correct, or at least not far off. Decide on a rule (e.g., if the number is within the range mentioned, count as correct).
- For **fields that were missing in the truth and supposed to be null**, check the model indeed gave you null or "N/A". If your ground truth for salary was null and the model hallucinated a number, that’s a false positive extraction – you should count that as an error.

**4. Simple Accuracy Metrics:** After comparing, you can tally up some metrics:
- **Per-field accuracy:** e.g., Name correct in 9/10 cases, Skills correct in 8/10, etc. This tells you which fields are easiest or hardest for the model.
- **Full record accuracy:** how many resumes got **all** fields correct. This is a stricter measure and will likely be lower. Even if one field is off, the whole record is off for this metric.
- You might also categorize errors (e.g., out of the errors: how many were because the model added info vs missed info vs misread info).

**5. Refinement Loop:** Just like any LLM task, use the insights from eval to improve your prompt or schema. If you notice the model often misunderstands the education level (maybe confusing diploma vs bachelor), consider tweaking the description for that field or adding an example in the system message. If it’s skipping certain skills, maybe those resumes had skills in a tricky format (like a comma-separated paragraph) that you could call out in the prompt (e.g., "skills may be listed in a comma-separated line, make sure to capture all of them").

Remember, our evaluation at this stage is relatively simple and mostly manual or heuristic. In a production scenario, you might develop a more sophisticated eval harness, but the principle is the same as what we did for SQL: start with a small, representative test set and gradually expand. Each time you adjust your extraction prompt or upgrade the model, you can rerun this eval set to see if things improved or if any new mistakes popped up.

---

### Practical Assignment: Resume Parser Pipeline

Now it’s your turn to apply these ideas. In this assignment, you’ll build a mini pipeline to extract structured data from resumes and evaluate its performance on a few examples. This will solidify your understanding of routing, schema design, and evaluation for extraction tasks.

**What to Do:**

1. **Prepare a Resume Dataset:** Get a collection of [resumes in text form](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data). For the sake of this exercise, 10–20 resumes is plenty.
2. **Define Your Schema:** Use Pydantic to define a `ResumeInfo` model (you can use the one we outlined above as a starting point). If you think of additional fields that might be interesting, feel free to include them, but don’t go overboard – focus on the core fields discussed.
3. **Implement the Extraction Loop:** Write a script or notebook that reads each resume from the dataset and calls the LLM to extract fields. Use the OpenAI API with structured output (function calling or `ChatCompletion.parse` with your Pydantic model). Make sure your prompt instructs the model properly (system message explaining the task, etc.). Collect the model’s output for each resume, and store it (perhaps as a JSON string or directly as a Pydantic object).
4. **Manual Annotation for Eval:** Pick **10 examples** (if your dataset is larger, or use all if you only made 10) and manually create the ground truth structured data for them. Essentially, read the resume and fill out what you believe the `ResumeInfo` should contain. Do this without looking at what the model output – pretend you’re the truth oracle. Save these expected results in a list or another CSV for comparison.
5. **Evaluate the Outputs:** Compare the model’s output to your manual annotations for those 10 resumes. You can write a simple function to compare each field:
    - For strings, you might do a case-insensitive exact match or check for containment if you expect minor phrasing differences.
    - For lists, you can compare as sets (order-agnostic).
    - For numbers, check equality or if within an acceptable range (if applicable).
    - For optional fields, ensure that if the truth is null, the model output is null (or at least not a completely wrong guess).
      For each resume, record which fields were correct and which were not.
6. **Report Your Findings:** Summarize how many of the 10 resumes were fully correct, and which fields tended to have errors. Did the model hallucinate any values that weren’t actually in the text? Did it miss information that was there? This analysis will help you pinpoint any weaknesses.
7. **(Optional) Tweak and Repeat:** If you have time, improve your prompt or schema based on the errors you saw, run the extraction again on those problem cases, and see if the accuracy improves. For instance, if `education_level` was often slightly off, maybe add a clarifying note in that field’s description or provide the model an example format in the prompt.

**Goals:**

- **Accurate Extraction:** Aim to correctly extract the majority of the fields for each resume in your test set. It’s okay if not 100% perfect (some resumes are tricky), but you want to see solid performance, especially on clearly stated fields like name or skills.
- **Robust to Variations:** Ensure your pipeline works on different resume styles. Maybe one resume lists skills in a bullet list, another in a paragraph; one has “M.Sc. in Computer Science” while another says “Master of Science (Computer Science)”. Your model should handle these variations – thanks to the power of the LLM and your field descriptions.
- **Graceful Handling of Missing Info:** Verify that when a resume truly lacks a piece of data, your system doesn’t break. The output should have `null` or a sensible placeholder for missing fields, and your evaluation should count that as **correct** (because that’s the correct behavior).
- **Evaluation Mindset:** Get comfortable with the idea of evaluating LLM outputs that aren’t just a single number or answer. You’re effectively writing test cases for a more complex output. This exercise should highlight the importance of well-chosen test examples and clear criteria for what counts as a correct extraction.

By completing this assignment, you will have built a mini "resume parser" powered by an LLM – a task that traditionally might require painstaking rule-based NLP or keyword searches. Even more, you’ll have a sense of how to evaluate and iteratively improve such a system, which is exactly the skill you need to tackle real business documents in your day-to-day work.

**Happy Extracting!** 🚀