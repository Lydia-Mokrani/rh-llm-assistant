# RH LLM Assistant

## 🎯 Objectif

Créer une IA pour aider les RH à :

* analyser un CV
* calculer le matching avec un poste
* donner des recommandations

---

## 📁 Structure du projet

* `dataset/` → données pour entraîner le modèle (Rania)
* `model/` → fine-tuning du LLM (Lydia)
* `api/` → API avec FastAPI (Kenza)

---

## ⚙️ Technologies

* Python
* Transformers
* QLoRA
* FastAPI

---

## 🚀 Tâches

### 👩 Rania (Dataset)

* créer `dataset/dataset.json`
* ajouter 1500+ exemples (input/output)

### 👩 Lydia (Modèle)

* fine-tuning avec QLoRA
* tester le modèle

### 👩 Kenza (API)

* créer API avec FastAPI
* endpoint `/analyze`

---

## ⏱️ Organisation

* travail en parallèle
* partager les avancées chaque jour
* tester régulièrement

---

## 📌 Format dataset (IMPORTANT)

```json
{
 "input": "CV + poste",
 "output": "score + analyse + recommandations"
}
```
