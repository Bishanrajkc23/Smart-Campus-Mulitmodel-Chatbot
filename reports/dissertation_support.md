# Dissertation Support: Smart Campus Tour & Information Multi-Modal Chatbot

## Introduction

This project presents a multimodal campus assistant designed to support students, staff, and visitors in retrieving location-specific campus information. The system accepts images of buildings, spoken queries, typed text questions, or combined multimodal input. It returns building identification, directions-oriented information, opening hours, events, facility descriptions, confidence scores, and ranked retrieval candidates.

The work follows an applied artificial intelligence architecture suitable for a time-constrained MSc project: CLIP is used for zero-shot visual retrieval, Whisper is used for speech recognition, DistilBERT is fine-tuned for intent classification, SentenceTransformers support semantic knowledge-base retrieval, and a lightweight neural fusion layer combines missing or present modalities.

## Dataset Design

The dataset is synthetic but structured to resemble a realistic university campus. It contains 20 campus locations, including learning spaces, teaching buildings, administrative services, wellbeing facilities, food services, creative spaces, research spaces, and accessibility support. Each record contains:

- Location name
- Category
- Natural-language description
- Opening hours
- Latitude and longitude coordinates
- Event list

The FAQ dataset contains at least 300 examples across 10 intent classes: `find_location`, `opening_hours`, `event_query`, `facility_information`, `study_space`, `food_services`, `accessibility`, `greeting`, `goodbye`, and `other`. Templates are used to ensure balanced coverage while maintaining controlled labels.

## Knowledge Base Design

The knowledge base is implemented in SQLite for reproducibility and simple local deployment. A normalized `campus_locations` table stores the structured location information, while an `interactions` table records user queries, active modalities, predicted locations, and confidence scores. A JSON export is also generated for transparent inspection and dissertation appendices.

SQLite is suitable for this project because the data volume is small, the schema is stable, and local deployment is required. The design can be extended to PostgreSQL or a managed vector database for larger campuses.

## Preprocessing

Image preprocessing resizes images to the CLIP input scale, converts them to RGB, and places them on a consistent canvas. Augmentation support includes random crop-style resizing, small rotations, contrast changes, and color jitter variants.

Audio preprocessing uses Librosa to load audio at 16 kHz, convert it to mono, compute MFCC features, normalize feature channels, and pad or truncate sequences to a fixed frame length.

Text preprocessing lowercases queries, removes punctuation, tokenizes by whitespace, removes a compact stopword list, and applies a lightweight suffix-based lemmatization fallback. HuggingFace tokenization is used for DistilBERT training and inference.

## Model Design

The system uses a modular design:

- Vision module: CLIP image-text retrieval.
- Audio module: Whisper speech-to-text and MFCC preprocessing.
- Text module: DistilBERT intent classifier.
- Retrieval module: SQLite plus SentenceTransformer semantic search.
- Fusion module: zero-vector padding, modality mask, concatenation, MLP scoring, and semantic retrieval blending.

This architecture avoids unnecessary from-scratch image model training and focuses project effort on integration, evaluation, and usability.

## CLIP Retrieval Architecture

Campus descriptions are converted into natural-language prompts and embedded using `openai/clip-vit-base-patch32`. Uploaded images are embedded with the CLIP image encoder. FAISS stores normalized text embeddings and performs inner-product similarity search, which is equivalent to cosine similarity for normalized vectors. The system returns the top-1 building prediction, similarity score, and top-3 candidates.

Evaluation uses top-1 accuracy, top-3 accuracy, and a confusion matrix over labelled synthetic campus signage images.

## Whisper Pipeline

The audio system accepts `wav`, `mp3`, and `m4a` files. Whisper transcribes the uploaded audio, and the transcript is passed to the same text and fusion pipeline used by typed queries. Speech recognition is evaluated using word error rate (WER), calculated as the edit distance between reference and predicted word sequences divided by the number of reference words.

## DistilBERT Intent Classifier

The text system fine-tunes `distilbert-base-uncased` for 10-way intent classification. The model returns an intent label and confidence score. Metrics include accuracy, weighted precision, weighted recall, weighted F1, confusion matrix, and a classification report.

The project also includes an interpretable rule-based fallback. This fallback is not intended to replace fine-tuning; it exists to keep the Streamlit demonstration usable if the model checkpoint has not yet been trained.

## Multimodal Fusion

The fusion model accepts three modality embeddings:

- CLIP image embedding
- Text query embedding
- Whisper transcript embedding

Missing modalities are represented by zero-vector padding. A binary mask indicates which modalities are present. The model concatenates the three embeddings and the mask, then feeds the result into an MLP that scores campus records. Scores are blended with semantic retrieval results from the knowledge base to improve robustness for sparse text queries.

This design supports image-only, voice-only, text-only, and combined multimodal input.

## Evaluation Results Template

The following tables should be populated after running `python -m src.evaluate` and `python -m src.train_intent`.

| Component | Metric | Result |
|---|---:|---:|
| Vision retrieval | Top-1 accuracy | To be generated |
| Vision retrieval | Top-3 accuracy | To be generated |
| Speech recognition | Mean WER | To be generated |
| Intent classifier | Accuracy | To be generated |
| Intent classifier | Weighted precision | To be generated |
| Intent classifier | Weighted recall | To be generated |
| Intent classifier | Weighted F1 | To be generated |
| Fusion retrieval | End-to-end accuracy | To be generated |
| Fusion retrieval | Top-3 retrieval accuracy | To be generated |

Figures are saved under `reports/figures`, and CSV or JSON metrics are saved under `reports/results`.

## Ethical Considerations

The project is designed around synthetic data to avoid collecting identifiable student data during development. If deployed with real campus images or speech recordings, explicit consent, retention limits, and secure storage would be required. Users should be informed that model predictions can be uncertain and should not be treated as authoritative in emergencies.

## GDPR Compliance

Potential personal data includes voice recordings, transcripts, uploaded images containing faces, and interaction logs. A GDPR-compliant deployment should implement:

- Data minimization
- Purpose limitation
- User consent
- Clear retention policy
- Deletion requests
- Encryption at rest and in transit
- Avoidance of unnecessary biometric processing

The current implementation stores only query text, modality labels, predicted location, confidence, and timestamp in the interaction log. Production use should add anonymization and configurable logging.

## Bias Analysis

The synthetic dataset may over-represent common campus buildings and under-represent edge cases such as temporary closures, renamed spaces, or accessibility routes. Whisper accuracy may vary across accents, noisy environments, and microphone quality. CLIP retrieval can be affected by visual similarity between buildings and by weak or ambiguous signage.

Mitigation strategies include collecting diverse evaluation samples, testing with international student accents, adding real campus imagery, and auditing errors by intent, location type, and modality.

## Limitations

The project does not perform true route optimization or indoor navigation. Image retrieval is based on CLIP similarity rather than a trained campus-specific classifier. The fusion model is lightweight and would require labelled multimodal interaction data for strong supervised performance. Synthetic data is appropriate for an MSc prototype but cannot fully substitute for real-world deployment testing.

## Future Work

Future work should add real campus photographs, multilingual support, GPS-aware route planning, indoor maps, live event APIs, opening-hour updates, human feedback correction, and a larger evaluation set. Fusion could be improved with contrastive multimodal training and calibrated uncertainty estimation.

## Harvard-style References

Johnson, J., Douze, M. and Jegou, H. (2019) 'Billion-scale similarity search with GPUs', IEEE Transactions on Big Data, 7(3), pp. 535-547.

Radford, A. et al. (2021) 'Learning transferable visual models from natural language supervision', Proceedings of the 38th International Conference on Machine Learning.

Radford, A. et al. (2022) 'Robust speech recognition via large-scale weak supervision'. Available at: https://arxiv.org/abs/2212.04356

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence embeddings using Siamese BERT-networks', Proceedings of EMNLP-IJCNLP, pp. 3982-3992.

Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019) 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter'. Available at: https://arxiv.org/abs/1910.01108

Wolf, T. et al. (2020) 'Transformers: State-of-the-art natural language processing', Proceedings of EMNLP: System Demonstrations, pp. 38-45.
