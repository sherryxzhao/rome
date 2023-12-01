## Processed Counterfact data link
- https://drive.google.com/file/d/1fFS6eMDwWky7qNYIeDqIjQK4igNljgBC/view?usp=sharing
- 
## Example of processed Counterfact data with max layer info
```json
[
  {
    "case_id": 0,
    "pararel_idx": 2796,
    "requested_rewrite": {
      "prompt": "The mother tongue of {} is",
      "relation_id": "P103",
      "target_new": {
        "str": "English",
        "id": "Q1860"
      },
      "target_true": {
        "str": "French",
        "id": "Q150"
      },
      "subject": "Danielle Darrieux",
      "max_edit_layer": 33
    },
    "paraphrase_prompts": [
      "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native",
      "An album was recorded for Capitol Nashville but never released. Danielle Darrieux spoke the language"
    ],
    "neighborhood_prompts": [
      "The mother tongue of L\u00e9on Blum is",
      "The native language of Montesquieu is",
      "Fran\u00e7ois Bayrou, a native",
      "The native language of Raymond Barre is",
      "Michel Rocard is a native speaker of",
      "Jacques Chaban-Delmas is a native speaker of",
      "The native language of Fran\u00e7ois Bayrou is",
      "Maurice Genevoix, speaker of",
      "The mother tongue of Fran\u00e7ois Bayrou is",
      "Melchior de Vog\u00fc\u00e9, speaker of"
    ],
    "attribute_prompts": [
      "J.\u00a0R.\u00a0R. Tolkien is a native speaker of",
      "The mother tongue of Douglas Adams is",
      "The mother tongue of Paul McCartney is",
      "Elvis Presley is a native speaker of",
      "Barack Obama, speaker of",
      "Douglas Adams, speaker of",
      "Meryl Streep, a native",
      "George Orwell spoke the language",
      "George Washington, a native",
      "Michael Jackson, a native"
    ],
    "generation_prompts": [
      "Danielle Darrieux's mother tongue is",
      "Where Danielle Darrieux is from, people speak the language of",
      "Danielle Darrieux's mother tongue is",
      "Danielle Darrieux was born in",
      "Danielle Darrieux's mother tongue is",
      "Danielle Darrieux's mother tongue is",
      "Danielle Darrieux was born in",
      "Where Danielle Darrieux is from, people speak the language of",
      "Danielle Darrieux was born in",
      "Danielle Darrieux was born in"
    ]
  },
]
```