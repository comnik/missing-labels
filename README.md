# Semi-supervised Image Classification

## Usage

General:

    python classify.py [caching options] [output mode]

By default, everything is loaded from snapshots in the snapshot/ subdirectory.
Caching options are **--reload** and **--reclassify**. Those trigger recomputing
input features and labels (and snapshotting them) or re-training the classifier respectiveley.

Output modes are **--cv** for cross-validation, **--validate** for generating validation-set predictions
and finally **--test** for generating test-set predictions. Predictions are stored in the out/ subdirectory.

So when executing classify.py for the first time:

    python classify.py --reload --reclassify --cv

Afterwards you can usually stop reloading:

    python classify.py --reclassify [output mode]

After finding the best classifier, you can quickly generate different outputs:

    python classify.py --cv
    python classify.py --validate
    python classify.py --test

