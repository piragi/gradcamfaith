# Description of the project
In this project, we analyze vision models (ViT's) and their explainability methods (current SotA: Hila Chefer's Attention LRP) in regards to faithfulness. Faithfulness in this project is measured through different measures, for instance the relative ordering of attribution value to their classification impact (SaCo score, On the Faithfulness of Vision Transformer Explanations), faithfulness correlation (Evaluating and aggregating feature-based model explanations) and PixelFlipping (On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation).
We employ sparse autencoders (SAEs) to to understand if we can find features that are inherently faithful, if they might be missed by current explainability methods and if they can be used to improve faithfulness.

This project is already quite advanced, generating a method to analyze features and rank them according to a score where attribution and classification impact is misaligned. But there need to be other questions answered first for this to be a comprehensive piece of research.
# Questions that need to be answered

## Why are SAE features themselves not used as an explanation?

## Why does simply taking topk activated features (during inference) and just use those for boosting attribution maps not work?

## Do the highly attributed regions correlate with highly active (SAE) features?

## Why do only some features contribute to faithfulness?

## Why are some features, which are actually important for faithfulness, constantly underrepresented?

## Why do we not use these faithful features for steering? Why do they not work in steering?

--- 

# Questions about the method itself
There are several design choices about this method which need to be reflected and justified. Measuring faithfulness is not a standardized operation.

## How do we define faithfulness in this project and why do we chose this definition?

## What does the method find exactly?
Features that are generally important for the attribution map but missed by current methods.
## How big does the analysis set need to be for the method to work?

## How many features are identified as faithfully misaligned?

## How many features are correctly identifed as being faithfully aligned?

## Are there differences between the different layers (early, middle, late)?

## How about feature interaction? Are there synergistic effects?

## Do the attribution maps get more humanly understandable?

## When does the method fail?

## Comparison with alternative explanations?

## How does the method compare to baseline (no boosting) and random feature selection?

## How do we validate that identified faithful features generalize beyond the analysis set?

## Are there differences in faithfulness results depending on SAE architecture and hyperparameters?

---

# Question Priority Ranking for Research

Based on the project's core objectives of using SAEs to improve faithfulness in vision transformer explanations, here is a ranked prioritization of all research questions:

## **IMPERATIVE QUESTIONS** (Must be answered for research validity)

### **Tier 1: Foundation Questions**
1. **"How do we define faithfulness in this project and why do we chose this definition?"** - **CRITICAL**
   - Without a clear, justified faithfulness definition, all subsequent analysis lacks theoretical grounding
   - Establishes the research framework and allows for reproducible evaluation

2. **"Why are SAE features themselves not used as an explanation?"** - **CRITICAL** 
   - Core theoretical question that justifies the entire research direction
   - If SAE features are inherently good explanations, the project's premise changes fundamentally

3. **"Do the highly attributed regions correlate with highly active (SAE) features?"** - **CRITICAL**
   - Establishes the relationship between current attribution methods and SAE decomposition
   - Essential baseline measurement for understanding the attribution-SAE feature gap
   - *Implementation: Spatial correlation analysis between attribution maps and SAE activation patterns*

3.1 **How do some features contribute to faithfulness?** - **CRITICAL**
   - Figure out the mechanism behind these features and why it improves faithfulness 

### **Tier 2: Method Validation Questions**
4. **"What does the method find exactly?"** - **HIGH**
   - Clearly defines the research contribution and expected outcomes
   - Critical for setting proper expectations and evaluation criteria

5. **"Why does simply taking topk activated features (during inference) and just use those for boosting attribution maps not work?"** - **HIGH**
   - Validates the need for the sophisticated feature selection approach
   - Rules out simpler alternatives that could undermine the method's complexity
   - *Implementation: Add "topk_activation" mode to build_boost_mask_improved() and compare faithfulness results*

6. **"Why do only some features contribute to faithfulness?"** - **HIGH**
   - Core insight needed to understand SAE feature heterogeneity in faithfulness tasks
   - Drives the feature selection methodology

## **HIGH PRIORITY QUESTIONS** (Significantly strengthen research)

7. **"Are there differences between the different layers (early, middle, late)?"** - **HIGH**
   - Layer-wise analysis provides crucial insights into where SAE features are most faithful
   - Aligns with prior work showing middle layers are crucial for visual processing
   - *Implementation: Group features by layer in saco_feature_analysis_simple.py and compare distributions*

8. **"How many features are identified as faithfully misaligned?"** - **HIGH**
   - Quantifies the scope of the problem and method's impact
   - Essential for understanding the practical significance
   - *Implementation: Add summary statistics to saco_feature_analysis_simple.py output*

9. **"When does the method fail?"** - **HIGH**
   - Critical for establishing method limitations and applicability bounds
   - Prevents overstatement of results and guides future improvements

10. **"Why are some features, which are actually important for faithfulness, constantly underrepresented?"** - **HIGH**
    - Addresses systematic biases in current attribution methods
    - Could reveal fundamental limitations in attention-based explanations

11. **"Why do we not use these faithful features for steering? Why do they not work in steering?"** - **HIGH**
    - Critical bridge between feature identification and practical application
    - Addresses the gap between faithfulness analysis and model control/improvement
    - Could reveal fundamental differences between features good for explanation vs. intervention
    - *Implementation: Test identified faithful features in build_boost_mask_improved() and measure steering effectiveness*

## **MEDIUM PRIORITY QUESTIONS** (Important for comprehensive understanding)

12. **"How many features are correctly identified as being faithfully aligned?"** - **MEDIUM**
    - Provides baseline performance metrics but is less critical than misalignment detection
    - Important for complete evaluation but not central to the research contribution
    - *Implementation: Count negative log_ratio features in existing saco_feature_analysis_simple.py*

13. **"How big does the analysis set need to be for the method to work?"** - **MEDIUM**
    - Practical consideration for method deployment but not core to the research question
    - Can be addressed through empirical analysis with varying dataset sizes
    - *Implementation: Run pipeline with different subset sizes and track result stability*

14. **"How about feature interaction? Are there synergistic effects?"** - **MEDIUM**
    - Interesting theoretical question but adds complexity without clear immediate benefit
    - Could be valuable for future work but not essential for proving the core concept

15. **"Comparison with alternative explanations?"** - **MEDIUM**
    - Important for positioning the work but not essential for validating the core hypothesis
    - Can be addressed through benchmarking against existing methods
    - *Implementation: Run faithfulness evaluation with different attribution methods (GradCAM, LRP variants) using existing pipeline*

16. **"How does the method compare to baseline (no boosting) and random feature selection?"** - **HIGH**
    - Essential for demonstrating that the method provides meaningful improvement over simple baselines
    - Critical for validating that sophisticated feature selection is necessary
    - *Implementation: Compare faithfulness scores between original attribution, random feature boosting, and method's feature boosting*

17. **"How do we validate that identified faithful features generalize beyond the analysis set?"** - **HIGH**
    - Critical for ensuring the method's findings are robust and not overfitted to specific data
    - Essential for establishing scientific validity and reproducibility
    - *Implementation: Train/val/test split methodology - identify features on train, validate on val, test on test set*

## **MEDIUM PRIORITY QUESTIONS** (Important for comprehensive understanding)

12. **"How many features are correctly identified as being faithfully aligned?"** - **MEDIUM**
    - Provides baseline performance metrics but is less critical than misalignment detection
    - Important for complete evaluation but not central to the research contribution
    - *Implementation: Count negative log_ratio features in existing saco_feature_analysis_simple.py*

13. **"How big does the analysis set need to be for the method to work?"** - **MEDIUM**
    - Practical consideration for method deployment but not core to the research question
    - Can be addressed through empirical analysis with varying dataset sizes
    - *Implementation: Run pipeline with different subset sizes and track result stability*

14. **"How about feature interaction? Are there synergistic effects?"** - **MEDIUM**
    - Interesting theoretical question but adds complexity without clear immediate benefit
    - Could be valuable for future work but not essential for proving the core concept

15. **"Comparison with alternative explanations?"** - **MEDIUM**
    - Important for positioning the work but not essential for validating the core hypothesis
    - Can be addressed through benchmarking against existing methods
    - *Implementation: Run faithfulness evaluation with different attribution methods (GradCAM, LRP variants) using existing pipeline*

18. **"Are there differences in faithfulness results depending on SAE architecture and hyperparameters?"** - **MEDIUM**
    - Important for understanding the generalizability of findings across different SAE configurations
    - Helps establish whether results are SAE-specific or represent general principles
    - *Implementation: Test with different SAE architectures/hyperparameters using existing pipeline*

## **LOWER PRIORITY QUESTIONS** (Nice to have)

19. **"Do the attribution maps get more humanly understandable?"** - **LOW**
    - While interesting, human evaluation is expensive and subjective
    - The faithfulness metrics already provide objective measures of explanation quality
    - Could be valuable for downstream applications but not core to the research

## **Rationale for Prioritization**

The ranking prioritizes questions that:
1. **Establish theoretical foundation** - Without clear definitions and justifications, the research lacks scientific rigor
2. **Validate core assumptions** - Questions that could invalidate the research premise must be answered first
3. **Quantify impact** - Understanding the scope and scale of the problem and solution
4. **Identify limitations** - Critical for honest scientific reporting and future work direction
5. **Provide mechanistic insights** - Understanding why certain features behave differently advances the field

**Questions ranked lower** either address implementation details that can be determined empirically, add complexity without clear immediate benefit, or focus on subjective evaluations that, while valuable, are not essential for proving the core scientific contribution.

## **Implementation Strategy Summary**

### **Questions Answerable with Current Codebase (Empirical):**
- Attribution-SAE correlation analysis
- Top-k activation baseline comparison  
- Layer-wise feature analysis
- Feature count statistics
- Dataset size sensitivity analysis
- Alternative explanation method comparison
- Faithful feature steering experiments
- Baseline comparison (no boosting vs random vs method)
- Validation methodology (train/val/test split evaluation)
- SAE architecture/hyperparameter sensitivity analysis

### **Questions Requiring Theoretical Analysis (Literature-based):**
- **"Why are SAE features themselves not used as an explanation?"** - Requires analysis of SAE literature and explanation theory
- **"What does the method find exactly?"** - Needs clear theoretical framework definition
- **"How do we define faithfulness and why?"** - Literature review of faithfulness definitions and justification
- **"Why do only some features contribute to faithfulness?"** - Theoretical understanding of feature properties
- **"Why are some features constantly underrepresented?"** - Analysis of attribution method limitations
- **"When does the method fail?"** - Theoretical bounds analysis combined with empirical failure cases
- **"Do attribution maps get more humanly understandable?"** - Requires human evaluation studies
