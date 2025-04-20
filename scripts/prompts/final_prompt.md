## Final prompt:

```
### System:

Gene Ontology (GO) is a widely used bioinformatics resource that provides a structured vocabulary for annotating and categorizing genes and gene products based on their biological functions, cellular locations, and molecular activities. You are a gene ontology expert, and your objective is to use your knowledge of the biological domain to write a response that appropriately completes the instruction.

### Instruction:

Use the input sentence below to label the tokens: terms, GO concepts and parents. A term is a word or sequence of words that MUST be present in the input sentence. Each term is associated with a concept in Gene Ontology hierarchy. A GO concept therefore refers to a specific category within the hierarchy. Each term can only be associated with one GO concept. However, a GO concept can have one or more parents. A parent represents immediate predecessor of a GO concept and is also a concept within GO hierarchy. The response SHOULD be in JSON format with a list of terms, GO concepts and parents. Terms and GO concepts CANNOT be a nested list but parents CAN be a nested list. Furthermore, the response SHOULD have equal number of terms, GO concepts and parents.

### Input:

Interestingly , AVPR2 - deficient male pups also die within the first week after birth.

### Response:

{
    "terms": [
        "die",
        "birth"
    ], 
    "GO concepts": [
        "death",
        "parturition"
    ],
    "parents": [
        [], 
        [
            "multi-multicellular organism process",
            "multi-organism reproductive process"
        ]
    ]
}
```