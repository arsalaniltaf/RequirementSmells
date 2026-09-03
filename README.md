# Explainable, Label-Correlation-Aware Multi-Label Requirement Smell Detection

This repository contains the implementation and experimental resources for the research project:

> **An XAI Framework for Multi-Label Requirement Smell Detection and AI-Assisted Requirement Revision**

The project investigates automated detection of requirement smells in natural language software requirements using multi-label machine learning approaches.

## Overview

The framework includes:

- Dataset harmonization and preprocessing
- Multi-label requirement smell detection
- Binary Relevance baseline
- Classifier Chains for label-dependency modelling
- Label co-occurrence analysis
- SHAP-based explainability
- AI-assisted requirement revision
- Reviewer-response experiments, including chain-order robustness and per-label evaluation

## Requirement Smell Labels

The framework detects five requirement smell categories:

1. Subjective
2. Ambiguous
3. Nonverifiable
4. Negative
5. Vague

## Project Structure

```text
RequirementSmells/
├── data/
├── src/
├── reviewer_response_experiments/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore