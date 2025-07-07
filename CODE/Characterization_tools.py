## Pseudocode for parameters analysis

Innovations: The CSWM integrates spatial sampling, geometric intersection calculation, and statistical normalization. During computation, a combination of endpoint detection and area-based weighting is utilized.

Circular window design: Adaptive spatial sampling
1. Aspect ratio analysis: Compare X and Y range dimensions//Determine optimal circle distribution strategy.
2. Radius calculation: Set circle radius based on shorter dimension//Ensure uniform sampling density.
3. Grid generation: Calculate circle center positions//Apply edge-aligned spacing for coverage.

Density computation: endpoint-based analysis
4. Endpoint detection: For each circle, count segment endpoints within radius//Apply Euclidean distance filtering.
5. Area normalization: Divide endpoint count by circle area//Apply geometric correction factor.

Intensity computation: intersection-based analysis
6. Intersection calculation: Count line-circle boundary crossings//Apply analytical geometry intersection algorithm.
7. Perimeter normalization: Divide intersection count by effective circle perimeter//Use standardized perimeter formula.

Matrix assembly: Spatial result organization
8. Density mapping: Store values in spatial grid//Handle coordinate transformation.
9. Intensity mapping: Store values in spatial grid//Maintain alignment with density results.

---

Note: The spatial sampling and intersection algorithms follow established computational geometry principles.
