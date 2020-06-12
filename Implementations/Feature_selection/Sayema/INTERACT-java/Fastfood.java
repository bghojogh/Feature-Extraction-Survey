/*
*    RELEASE INFORMATION (Nov. 06, 2005)
*
*    Normal Frame work for combine search and metric and help to
*    implement algrithm fast:
*      Developped for Weka by Zheng Alan Zhao
*      Nov. 21, 2005
*
*    DESCRIPTION
*
*
*    INSTALL NOTES
*
*
*    CONTACT INFORMATION
*
*    Data Mining and Machine Learning Lab
*    Computer Science and Engineering Department
*    Fulton School of Engineering
*    Arizona State University
*    Tempe, AZ 85287
*
*    Fastfood.java
*
*    Copyright (C) 2004 Data Mining and Machine Learning Lab,
*                       Computer Science and Engineering Department,
*                       Fulton School of Engineering,
*                       Arizona State University
*    Valid options are: <p>
*
*    -P <start set> <br>
*    Specify a starting set of attributes. Eg 1,4,7-9. <p>
*
*    -T <threshold> <br>
*    Specify a threshold by which the AttributeSelection module can. <br>
*    discard attributes. <p>
*
*    -F <filename> <br>
*    Specify the file name of the dataset
*/

/*
 *    Fastfood.java
 *
 *
 */


package  weka.attributeSelection;

import  java.io.*;
import  java.util.*;
import  weka.core.*;

public class Fastfood extends ASSearch
  implements RankedOutputSearch, StartSetHandler, OptionHandler {

  /** Holds the starting set as an array of attributes */
  public int[] m_starting;

  /** Holds the start set for the search as a range */
  public Range m_startRange;

  /** Holds the ordered list of attributes */
  public int[] m_attributeList;

  /** Holds the list of attribute merit scores */
  public double[] m_attributeMerit;

  /** Data has class attribute---if unsupervised evaluator then no class */
  public boolean m_hasClass;

  /** Class index of the data if supervised evaluator */
  public int m_classIndex;

  /** The number of attribtes */
  public int m_numAttribs;

  /** Whether rank the features */
  public boolean m_doRank;

  /**
   * A threshold by which to discard attributes---used by the
   * AttributeSelection module
   */
  public double m_threshold = 0;

  /** The number of attributes to select. -1 indicates that all attributes
      are to be retained. Has precedence over m_threshold */
  public int m_numToSelect = -1;

  /** Used to compute the number to select */
  public int m_calculatedNumToSelect = -1;

  /** The file name of the dataset */
  public String m_fileName = null;
  
  /** training instances */
  public Instances m_trainInstances;


  /**
   * Returns a string describing this search method
   * @return a description of the search suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Fastfood based algotithm : \n\nCombination of Search and Metric. ";
  }

  /**
   * Constructor
   */
  public Fastfood () {
    resetOptions();
  }

  /**
   * Get inputs from the command line and put
   * data from data file to Instances Object
   * @param args String[] the inputs from the command line
   */
  public Fastfood (String[] args) {

    try{
      setOptions(args);
    }
    catch (Exception e)
    {
      System.out.print("There is an error occures when initial args\n"+e.toString());
      System.exit(9);
    }

    if (m_fileName == null){
      System.out.print("Please specify the file that contains the dataset");
      System.exit(9);
    }
    else {
      try  {
        System.out.print("\nLoading Data into Instances class...");
        FileReader file=new FileReader(m_fileName);
        m_trainInstances=new Instances(file);
        m_trainInstances.setClassIndex(m_trainInstances.numAttributes() -1);

        file.close();
        System.out.print(" finished.\n");
      }
      catch(Exception e)
      {
        System.out.print("Error occurs when initial data container\n"+e.toString());
        System.exit(9);
      }
    }

  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numToSelectTipText() {
    return "Specify the number of attributes to retain. The default value "
      +"(-1) indicates that all attributes are to be retained. Use either "
      +"this option or a threshold to reduce the attribute set.";
  }

  /**
   * Specify the number of attributes to select from the ranked list. -1
   * indicates that all attributes are to be retained.
   * @param n the number of attributes to retain
   */
  public void setNumToSelect(int n) {
    m_numToSelect = n;
  }

  /**
   * Gets the number of attributes to be retained.
   * @return the number of attributes to retain
   */
  public int getNumToSelect() {
    return m_numToSelect;
  }

  /**
   * Gets the calculated number to select. This might be computed
   * from a threshold, or if < 0 is set as the number to select then
   * it is set to the number of attributes in the (transformed) data.
   * @return the calculated number of attributes to select
   */
  public int getCalculatedNumToSelect() {
    if (m_numToSelect >= 0) {
      m_calculatedNumToSelect = m_numToSelect;
    }
    return m_calculatedNumToSelect;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String thresholdTipText() {
    return "Set threshold by which attributes can be discarded. Default value "
      + "results in no attributes being discarded. Use either this option or "
      +"numToSelect to reduce the attribute set.";
  }

  /**
   * Set the threshold by which the AttributeSelection module can discard
   * attributes.
   * @param threshold the threshold.
   */
  public void setThreshold(double threshold) {
    m_threshold = threshold;
  }

  /**
   * Returns the threshold so that the AttributeSelection module can
   * discard attributes from the ranking.
   */
  public double getThreshold() {
    return m_threshold;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String generateRankingTipText() {
    return "Whether we rank the features based on their contribution. ";
  }

  /**
   * This is a dummy set method---Ranker is ONLY capable of producing
   * a ranked list of attributes for attribute evaluators.
   * @param doRank this parameter is N/A and is ignored
   */
  public void setGenerateRanking(boolean doRank) {
    m_doRank = doRank;
  }

  /**
   * This is a dummy method. Ranker can ONLY be used with attribute
   * evaluators and as such can only produce a ranked list of attributes
   * @return true all the time.
   */
  public boolean getGenerateRanking() {
    return m_doRank;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String startSetTipText() {
    return "Specify a set of attributes to be included any way. "
      +" When generating the ranking, Ranker will not evaluate the attributes "
      +" in this list. "
      +"This is specified as a comma "
      +"seperated list off attribute indexes starting at 1. It can include "
      +"ranges. Eg. 1,2,5-9,17.";
  }

  /**
   * Sets a starting set of attributes for the search. It is the
   * search method's responsibility to report this start set (if any)
   * in its toString() method.
   * @param startSet a string containing a list of attributes (and or ranges),
   * eg. 1,2,6,10-15.
   * @exception Exception if start set can't be set.
   */
  public void setStartSet (String startSet) throws Exception {
    m_startRange.setRanges(startSet);
  }

  /**
   * Returns a list of attributes (and or attribute ranges) as a String
   * @return a list of attributes (and or attribute ranges)
   */
  public String getStartSet () {
    return m_startRange.getRanges();
  }

  public void setFileName (String fileName) {
    m_fileName = fileName;
  }

  public String getFileName () {
    return m_fileName;
  }

  /**
   * Returns an enumeration describing the available options.
   * @return an enumeration of all the available options.
   **/
  public Enumeration listOptions () {
    Vector newVector = new Vector(3);

    newVector
      .addElement(new Option("\tSpecify a starting set of attributes."
                             + "\n\tEg. 1,3,5-7."
                             +"\t\nAny starting attributes specified are"
                             +"\t\nincluded any way during the ranking."
                             ,"P",1
                             , "-P <start set>"));
    newVector
      .addElement(new Option("\tSpecify a theshold by which attributes"
                             + "\tmay be discarded from the ranking.","T",1
                             , "-T <threshold>"));

    newVector
      .addElement(new Option("\tSpecify number of attributes to select"
                             ,"N",1
                             , "-N <num to select>"));

  newVector
    .addElement(new Option("\tSpecify file name of dataset"
                           ,"F",1
                           , "-F <num to select>"));

    return newVector.elements();

  }

  /**
   * Parses a given list of options.
   *
   * Valid options are: <p>
   *
   * -P <start set> <br>
   * Specify a starting set of attributes. Eg 1,4,7-9. <p>
   *
   * -T <threshold> <br>
   * Specify a threshold by which the AttributeSelection module can <br>
   * discard attributes. <p>
   *
   * -N <number to retain> <br>
   * Specify the number of attributes to retain. Overides any threshold. <br>
   * <p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   *
   **/
  public void setOptions (String[] options)
    throws Exception {
    String optionString;
    resetOptions();

    optionString = Utils.getOption('P', options);
    if (optionString.length() != 0) {
      setStartSet(optionString);
    }

    optionString = Utils.getOption('T', options);
    if (optionString.length() != 0) {
      Double temp;
      temp = Double.valueOf(optionString);
      setThreshold(temp.doubleValue());
    }

    optionString = Utils.getOption('N', options);
    if (optionString.length() != 0) {
      setNumToSelect(Integer.parseInt(optionString));
    }

    optionString = Utils.getOption('F', options);
    if (optionString.length() != 0) {
      setFileName(optionString);
    }

  }

  /**
   * Gets the current settings of ReliefFAttributeEval.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions () {
    String[] options = new String[8];
    int current = 0;

    if (!(getStartSet().equals(""))) {
      options[current++] = "-P";
      options[current++] = ""+startSetToString();
    }

    options[current++] = "-T";
    options[current++] = "" + getThreshold();

    options[current++] = "-N";
    options[current++] = ""+getNumToSelect();

    options[current++] = "-F";
    options[current++] = ""+getFileName();

    while (current < options.length) {
      options[current++] = "";
    }
    return  options;
  }

  /**
   * converts the array of starting attributes to a string. This is
   * used by getOptions to return the actual attributes specified
   * as the starting set. This is better than using m_startRanges.getRanges()
   * as the same start set can be specified in different ways from the
   * command line---eg 1,2,3 == 1-3. This is to ensure that stuff that
   * is stored in a database is comparable.
   * @return a comma seperated list of individual attribute numbers as a String
   */
  protected String startSetToString() {
    StringBuffer FString = new StringBuffer();
    boolean didPrint;

    if (m_starting == null) {
      return getStartSet();
    }

    for (int i = 0; i < m_starting.length; i++) {
      didPrint = false;

      if ((m_hasClass == false) ||
          (m_hasClass == true && i != m_classIndex)) {
        FString.append((m_starting[i] + 1));
        didPrint = true;
      }

      if (i == (m_starting.length - 1)) {
        FString.append("");
      }
      else {
        if (didPrint) {
          FString.append(",");
        }
      }
    }

    return FString.toString();
  }

  /**
   * Kind of a dummy search algorithm. Calls a Attribute evaluator to
   * evaluate each attribute not included in the startSet and then sorts
   * them to produce a ranked list of attributes.
   *
   * @param ASEvaluator the attribute evaluator to guide the search
   * @param data the training instances.
   * @return an array (not necessarily ordered) of selected attribute indexes
   * @exception Exception if the search can't be completed
   * MUST BE OVERRIDE
   */
  public int[] search (ASEvaluation ASEval, Instances data)
    throws Exception {
    int [] rank = new int [1] ;
    return rank;

  }


  /**
   * Sorts the evaluated attribute list
   *
   * @return an array of sorted (highest eval to lowest) attribute indexes
   * @exception Exception of sorting can't be done.
   * MUST BE OVERRIDE
   */
  public double[][] rankedAttributes ()
    throws Exception {
    double [] [] rank = new double [1] [1];
    return rank;
  }

  protected void determineNumToSelectFromThreshold(double [][] ranking) {
    int count = 0;
    for (int i = 0; i < ranking.length; i++) {
      if (ranking[i][1] > m_threshold) {
        count++;
      }
    }
    m_calculatedNumToSelect = count;
  }

  protected void determineThreshFromNumToSelect(double [][] ranking)
    throws Exception {
    if (m_numToSelect > ranking.length) {
      throw new Exception("More attributes requested than exist in the data");
    }

    if (m_numToSelect == ranking.length) {
      return;
    }

    m_threshold = (ranking[m_numToSelect-1][1] +
                   ranking[m_numToSelect][1]) / 2.0;
  }

  /**
   * returns a description of the search as a String
   * @return a description of the search
   */
  public String toString () {
    StringBuffer BfString = new StringBuffer();
    BfString.append("\tFastfood Search.\n");

    if (m_starting != null) {
      BfString.append("\tAlways included attributes: ");

      BfString.append(startSetToString());
      BfString.append("\n");
    }

    if (m_threshold != -Double.MAX_VALUE) {
      BfString.append("\tThreshold for discarding attributes: "
                      + Utils.doubleToString(m_threshold,8,4)+"\n");
    }

    return BfString.toString();
  }


  /**
   * Resets stuff to default values
   */
  public void resetOptions () {
    m_starting = null;
    m_startRange = new Range();
    m_attributeList = null;
    m_attributeMerit = null;
    m_threshold = -Double.MAX_VALUE;
    m_fileName = null;
    m_doRank = false;
  }


  protected boolean inStarting (int feat) {
    // omit the class from the evaluation
    if ((m_hasClass == true) && (feat == m_classIndex)) {
      return  false;
    }

    if (m_starting == null) {
      return  false;
    }

    for (int i = 0; i < m_starting.length; i++) {
      if (m_starting[i] == feat) {
        return  true;
      }
    }

    return  false;
  }

}

