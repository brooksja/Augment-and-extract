# Augment-and-extract
Scripts to augment (initially) histopathology tiles and then extract features.

## Usage
1. Clone repo

2. Create environment from requirements.txt

3. Download relevant model weights (see below)

4. Navigate to the Augment-and-Extract folder

5. Run the following command:

		python -m src.main \
		extractor \
		/PATH/TO/CHECKPOINT \
		/PATH/TO/DATA \
		/PATH/FOR/OUTPUT \
		--repetitions INT \
		--extra_info
     
	extractor can be 'xiyue' or 'ozanciga', default is xiyue
  
	--repetitions is optional and defaults to 0
	
	-x is an optional flag to specify adding additional data from the clini table to the features. If specified, the user will then be asked to provide a clini table, a slide table and the column names to be looked at
  
## Weights
xiyue: https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL

ozanciga: https://github.com/ozanciga/self-supervised-histopathology/releases/tag/nativetenpercent

## Acknowledgments
Feature extractors based on marugoto by Kather AI (https://github.com/KatherLab/marugoto)

Ret_CCL from Wang et al. RetCCL: Clustering-guided Contrastive Learning for Whole-slide Image Retrieval (under review as of 18/10/22) 

Ozanciga extractor from Ozan Ciga, Tony Xu and Anne Louise Martel, Self supervised contrastive learning for digital histopathology, Machine Learning with Applications 7, 100198, 2022 (https://github.com/ozanciga/self-supervised-histopathology)
