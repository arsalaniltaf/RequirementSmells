# Explainable, Label-Correlation-Aware Multi-Label Requirement Smell Detection

This repository contains the implementation and experimental resources for the research project:

> **An XAI Framework for Multi-Label Requirement Smell Detection and AI-Assisted Requirement Revision**

The project investigates automated detection of requirement smells in natural language software requirements using multi-label machine learning approaches.

---

## Overview

Software requirements written in natural language may contain quality issues, commonly referred to as **requirement smells**. These issues can negatively affect requirement clarity, verification, interpretation, and subsequent software development activities.

This project investigates automated detection of multiple requirement smells simultaneously using machine learning and label-dependency-aware approaches.

The framework includes:

- Dataset harmonization and preprocessing
- Multi-label requirement smell detection
- Binary Relevance baseline
- Classifier Chains for label-dependency modelling
- Label co-occurrence analysis
- SHAP-based explainability
- AI-assisted requirement revision
- Reviewer-response experiments, including chain-order robustness and per-label evaluation

---

## Requirement Smell Labels

The framework detects five requirement smell categories:

1. **Subjective**
2. **Ambiguous**
3. **Nonverifiable**
4. **Negative**
5. **Vague**

A single requirement may contain more than one smell. Therefore, the task is formulated as a **multi-label classification problem**.

---

## Research Objectives

The project investigates the following aspects:

- How effectively can multiple requirement smells be detected simultaneously?
- How does Binary Relevance perform as a multi-label baseline?
- Can Classifier Chains benefit from dependencies between requirement smell labels?
- How do different classifier-chain orders affect performance?
- Can label co-occurrence information be incorporated into classifier-chain modelling?
- Which textual features contribute most to individual predictions?
- How can explainability support AI-assisted requirement revision?

---

## Methodology

### 1. Dataset Harmonization

Multiple requirement smell datasets are harmonized into a unified dataset.

The processing pipeline includes:

- Dataset merging
- Label harmonization
- Requirement cleaning
- Duplicate analysis
- Dataset validation
- Multi-label statistical analysis

The final harmonized dataset contains five unified smell labels:

```text
Subjective
Ambiguous
Nonverifiable
Negative
Vague
