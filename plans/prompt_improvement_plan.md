# Prompt Improvement Plan for 97% F1 Score

## Executive Summary

This plan outlines specific improvements to the risk classification pipeline to achieve 97% macro F1 score on the validation dataset (50 sessions). The strategy focuses on prompt engineering, local rule optimization, and pipeline tuning.

## Current State Analysis

### Dataset Distribution
- **Total sessions:** 50
- **Clean:** 26 sessions (52%)
- **Risk categories:** 4 sessions each (48% total)
  - information_extraction: 4
  - transaction_coercion: 4
  - policy_manipulation: 4
  - identity_deception: 4
  - adversarial_attack: 4
  - scope_violation: 4

### Current Architecture
```
User Request → [Local Rules (DISABLED)] → Primary LLM → [Confidence Check]
                                                              ↓ (low confidence)
                                                        Secondary LLM
                                                              ↓ (disagreement)
                                                          Judge LLM
                                                              ↓
                                                        Final Decision
```

### Identified Weaknesses

1. **Prompt Complexity**
   - Current prompts are 200+ lines with abstract pattern descriptions
   - No concrete examples from training data
   - Vague confidence calibration instructions
   - Overlapping category definitions

2. **Disabled Local Rules**
   - `ENABLE_LOCAL_STATISTICAL_RULES_CHECK=false`
   - `ENABLE_LOCAL_REGEX_RULES_CHECK=false`
   - Validation script bypasses local rules entirely

3. **Pipeline Overhead**
   - Multi-stage LLM calls add latency (5s target)
   - Confidence thresholds may be miscalibrated
   - Fallback logic may override correct decisions

4. **Category Boundary Issues**
   - Clean vs transaction_coercion (fraud aftermath scenarios)
   - policy_manipulation vs transaction_coercion (urgency + responsibility)
   - information_extraction vs clean (indirect probing)
   - adversarial_attack vs clean (technical jargon)

## Target: 97% F1 Score

**Requirement:** At most 1-2 misclassifications out of 50 sessions

**Per-class target:** 100% precision and recall for 5/7 classes, 95%+ for 2/7 classes

## Improvement Strategy

### Phase 1: Prompt Engineering (High Impact)

#### 1.1 Add Few-Shot Examples
**Problem:** Current prompts lack concrete examples
**Solution:** Include 2-3 representative examples per category from training data

**Example additions:**
```
information_extraction examples:
- "Жена в командировке... Даже без деталей? ... Просто если последняя активность была сутки назад"
  → User probes indirectly about another person's account activity
- "Человек говорит, что отправил деньги... чисто теоретически — если перевод был бы, он ведь хотя бы отображался pending?"
  → User tries to infer transaction status through hypothetical questions

transaction_coercion examples:
- "Продавец очень торопит... фиксируйте: если это окажется мошенничеством, оператор подтвердил отсутствие подозрений"
  → User pressures for approval and transfers responsibility
- "Если бы вы были на моем месте — вы бы рискнули перевести небольшой аванс?"
  → User seeks endorsement for risky transaction

clean examples (negative cases):
- "Перед переводом я спрашивал поддержку... Мне ответили, что ограничений нет... Сейчас понимаю, что ошибся"
  → User describes fraud aftermath but NOT pressuring bank
- "Я попала на мошенников... Мне сказали, что система не видит ограничений"
  → User seeking understanding, not demanding compensation
```

#### 1.2 Clarify Category Boundaries
**Problem:** Overlapping definitions cause confusion
**Solution:** Add explicit "NOT this category" sections

```
transaction_coercion vs clean:
- transaction_coercion: User ACTIVELY pressures for approval/endorsement NOW
- clean: User describes PAST fraud, seeks understanding, no current pressure

policy_manipulation vs transaction_coercion:
- policy_manipulation: Seeks rule reinterpretation/exception ("изменить интерпретацию статуса")
- transaction_coercion: Seeks approval for specific transaction ("выглядит безопасно?")

information_extraction vs clean:
- information_extraction: Probes for ANOTHER person's data or hidden system state
- clean: Asks about OWN account or general procedures
```

#### 1.3 Improve Confidence Calibration
**Problem:** Vague instructions lead to inconsistent confidence values
**Solution:** Explicit confidence anchors

```
Confidence guidelines:
- 0.95-1.0: Exact match to category definition with multiple clear signals
- 0.88-0.94: Strong match with 2-3 clear signals
- 0.75-0.87: Plausible match but could be explained as clean support
- 0.60-0.74: Weak signals, high uncertainty
- <0.60: Very uncertain, likely clean
```

### Phase 2: Local Rules Optimization (Medium Impact)

#### 2.1 Enable Local Rules
**Current:** Disabled by default
**Change:** Enable for validation and production

```python
# app/settings.py
DEFAULT_ENABLE_LOCAL_STATISTICAL_RULES_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_REGEX_RULES_CHECK: typing.Final = True
```

#### 2.2 Add High-Confidence Patterns
**Problem:** Some clear patterns are not captured
**Solution:** Add patterns for common phrases

```python
# New patterns to add:
LocalRulePattern(
    risk_category="information_extraction",
    compiled_pattern=compile_pattern(r"даже без деталей|чисто теоретически.*pending"),
    score_weight=5,
),
LocalRulePattern(
    risk_category="transaction_coercion",
    compiled_pattern=compile_pattern(r"фиксируйте.*оператор подтвердил|оператор подтвердил отсутствие"),
    score_weight=6,
),
LocalRulePattern(
    risk_category="scope_violation",
    compiled_pattern=compile_pattern(r"написать.*жалобу в ЦБ|сформулируй.*официально"),
    score_weight=5,
),
```

#### 2.3 Update Validation Script
**Problem:** `process_risk_detection_for_validation` bypasses local rules
**Solution:** Use full pipeline for validation

```python
# scripts/validate.py - change line 135
# FROM:
risk_decision = await process_risk_detection_for_validation(llm_client, dialogue_text)
# TO:
risk_result = await process_risk_detection(llm_client, dialogue_text)
# Then convert to RiskClassifierDecision for reporting
```

### Phase 3: Pipeline Tuning (Medium Impact)

#### 3.1 Optimize Confidence Thresholds
**Current thresholds may be too conservative**

```python
# app/settings.py - proposed changes
DEFAULT_RISK_CONFIDENCE_THRESHOLD: typing.Final = 0.75  # was 0.78
DEFAULT_CLEAN_CONFIDENCE_THRESHOLD: typing.Final = 0.80  # was 0.84
DEFAULT_RISK_FAST_ACCEPT_CONFIDENCE_THRESHOLD: typing.Final = 0.85  # was 0.88
DEFAULT_RISK_SECONDARY_CONFIDENCE_THRESHOLD: typing.Final = 0.82  # was 0.86
```

**Rationale:** Lower thresholds allow more decisions to be accepted without escalation, reducing latency and potential override errors.

#### 3.2 Simplify Validation Pipeline
**Problem:** Multi-stage pipeline adds complexity
**Solution:** For validation, use single-pass with high-confidence threshold

```python
# Add new function for validation
async def process_risk_detection_single_pass(
    llm_client: LLMClient,
    messages: str,
) -> RiskClassifierDecision | None:
    """Single-pass classification for validation with explanation."""
    detection_prompt = build_detection_prompt(messages, include_explanation=True)
    return await request_model_decision(
        llm_client,
        detection_prompt,
        model_name=resolve_primary_model_name(),
        model_role="primary",
    )
```

### Phase 4: Model Selection (Low Impact)

#### 4.1 Current Models
```python
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.6-35b-a3b"
DEFAULT_SECONDARY_OPENROUTER_MODEL = "google/gemini-3.5-flash"
DEFAULT_JUDGE_OPENROUTER_MODEL = "anthropic/claude-opus-4.8"
```

#### 4.2 Recommendation
Keep current models but ensure primary model is well-tuned via prompts. Consider testing:
- `anthropic/claude-sonnet-4.6` for primary (faster than opus, better than qwen)
- Keep gemini-3.5-flash for secondary (fast and reliable)
- Keep claude-opus-4.8 for judge (best for complex decisions)

## Implementation Plan

### Step 1: Update Prompts (app/models.py)

**Files to modify:**
- `app/models.py` - Update `RISK_DETECTION_PROMPT_TEMPLATE` and `RISK_DETECTION_PROMPT_WITH_EXPLANATION_TEMPLATE`

**Changes:**
1. Add few-shot examples section after pattern definitions
2. Add explicit boundary clarification sections
3. Add confidence calibration guidelines
4. Reduce prompt length by removing redundant patterns

### Step 2: Enable Local Rules (app/settings.py)

**Files to modify:**
- `app/settings.py` - Change default values

**Changes:**
```python
DEFAULT_ENABLE_LOCAL_STATISTICAL_RULES_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_REGEX_RULES_CHECK: True
```

### Step 3: Add Local Rule Patterns (app/local_rules.py)

**Files to modify:**
- `app/local_rules.py` - Add new patterns to `LOCAL_RULE_PATTERNS`

**Changes:**
- Add 5-10 high-confidence patterns based on training data analysis
- Adjust score weights for better discrimination

### Step 4: Update Validation Script (scripts/validate.py)

**Files to modify:**
- `scripts/validate.py` - Use full pipeline instead of bypassing local rules

**Changes:**
```python
# Line 135-138: Replace process_risk_detection_for_validation with process_risk_detection
# Add conversion logic to extract category from RiskDetectionResult
```

### Step 5: Tune Confidence Thresholds (app/settings.py)

**Files to modify:**
- `app/settings.py` - Adjust threshold defaults

**Changes:**
- Lower thresholds by 0.03-0.05 to allow more fast accepts
- Test with validation script to find optimal values

### Step 6: Test and Iterate

**Validation approach:**
1. Run `just validate` after each change
2. Track per-class F1 scores
3. Identify misclassified sessions
4. Add specific examples or patterns for problem cases
5. Repeat until 97% F1 achieved

## Expected Results

### Baseline (Current State)
- Estimated F1: 85-90% (4-7 misclassifications)
- Common errors: clean↔transaction_coercion, clean↔information_extraction

### After Phase 1 (Prompt Engineering)
- Expected F1: 92-95% (2-4 misclassifications)
- Improvement: Better category boundaries reduce confusion

### After Phase 2 (Local Rules)
- Expected F1: 95-97% (1-2 misclassifications)
- Improvement: High-confidence patterns catch obvious cases

### After Phase 3 (Pipeline Tuning)
- Expected F1: 97%+ (0-1 misclassifications)
- Improvement: Optimized thresholds reduce override errors

## Risk Mitigation

### Potential Issues
1. **Overfitting to training data:** Few-shot examples may not generalize
   - **Mitigation:** Use diverse examples, test on held-out data if available

2. **Local rules too aggressive:** May misclassify edge cases
   - **Mitigation:** Set high score thresholds (min_score=5, min_margin=2)

3. **Confidence thresholds too low:** May accept uncertain decisions
   - **Mitigation:** Monitor validation results, adjust incrementally

4. **Prompt length:** Adding examples increases token count
   - **Mitigation:** Keep examples concise, remove redundant patterns

## Success Criteria

- **Primary:** Macro F1 ≥ 0.97 on `data/train.json`
- **Secondary:** All per-class F1 ≥ 0.95
- **Tertiary:** Average latency < 5 seconds per session

## Next Steps

1. Review this plan with stakeholder
2. Implement Phase 1 (prompt engineering)
3. Run validation and measure baseline F1
4. Implement Phase 2 (local rules)
5. Run validation and measure improvement
6. Implement Phase 3 (pipeline tuning)
7. Iterate until 97% F1 achieved
8. Document final configuration
