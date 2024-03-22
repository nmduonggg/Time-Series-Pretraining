# ECG Dataset Processing

## [Will Two Do datasets](https://moody-challenge.physionet.org/2021/)

The electrocardiogram (ECG) is a non-invasive representation of the electrical activity of the heart. In 2017 we challenged the public to classify AF from a single-lead ECG, and in 2020 we challenged the public to diagnose a much larger number of cardiac problems using twelve-lead recordings. However, there is limited evidence to demonstrate the utility of reduced-lead ECGs for capturing a wide range of diagnostic information.

We are asking you to build an algorithm that can classify cardiac abnormalities from twelve-lead, six-lead, four-lead, three-lead, and two-lead ECGs.

The training data contains twelve-lead ECGs. The validation and test data contains twelve-lead, six-lead, four-lead, three-lead, and two-lead ECGs:

1. Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
2. Six leads: I, II, III, aVR, aVL, aVF
3. Four leads: I, II, III, V2
4. Three leads: I, II, V2
5. Two leads: I, II