RELEASE INFORMATION

INTERACT algorithm is a feature selection method based on Inconsistency and Symmetrical Uncertainty Measurement for feature interaction analysis. You can find the detail of INTERACT algorithm in:
Zheng Zhao, Huan Liu, Searching for Interacting Features. In Proc. of International Joint Conferences on Artificial Intelligence (IJCAI) 2007

INSTALL NOTES

1. Put INTERACT.class, Fastfood.class and INTERACT$hashKey.class into the directory: weak-class-home\attributeSelection\

2. Modify weak-class-home\gui\GenericObjectEditor.props:
	a) In section ¡°# Lists the Attribute Selection Search methods I want to choose from¡±
	    Add ¡°, \¡± to the last line, then add ¡°weka.attributeSelection.INTERACT¡± to the end of this section as a new line.

3. Run: java weka.gui.GUIChooser
	In Attribute Evaluator: You can select any evaluator. INTERACT use its own evaluator, which is embedded in the algorithm. So this option never takes effect.
	In Search Method: Select INTERACT

CONTACT INFORMATION

For the algorithm:
Huan Liu: hliu@asu.edu
Zheng Zhao : zhaozheng@asu.edu

For algorithm implementation:
Zheng Zhao: zhaozheng@asu.edu
 


