Extract detailed medical event information from clinical text. Identify the following event types:
- 'Adverse_event': An undesirable medical occurrence after the administration of a pharmaceutical product.
- 'Potential_therapeutic_event': A medical occurrence suggesting potential benefit after the administration of a pharmaceutical product.

For each event, identify the detailed attributes for 'Subject' and 'Treatment':

1. **Subject attributes**:
   - {{subject_desc}}

2. **Treatment attributes**:
   - {{treatment_desc}}

Output format should be a list of dictionaries. Use 'null' for missing values.
Format: {{sub_format_example}}

{{few_shot_examples}}
Text: {{sentence}}
Events: 
