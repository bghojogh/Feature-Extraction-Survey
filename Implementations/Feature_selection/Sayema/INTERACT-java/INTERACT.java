package weka.attributeSelection;

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.filters.supervised.attribute.Discretize;
import  weka.filters.Filter;


/*
*    RELEASE INFORMATION (Nov. 11, 2006)
*
*      INTERACT feature selection algorithm:
*      Developped for Weka by Zheng (Alan) Zhao
*      Nov. 11, 2006
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
*    INTERACT.java
*
*    Copyright (C) 2006 Data Mining and Machine Learning Lab,
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
*/


public class INTERACT extends Fastfood{


  /** number of instances in the training data */
  private int m_numInstances;

  /** Discretise numeric attributes */
  private Discretize m_disTransform;

  /** Hash table for evaluating feature subsets */
  private Hashtable m_table;

  /** Hash table for evaluating feature subsets */
  private Hashtable m_tmpTable;

  /** current features in the hash table */
  private boolean [] m_currentFeatureSet;

  /** features must be included in the selected feature set */
  private boolean [] m_mustIncludedFeatures;

  /** Inconsistency rate of the whole dataset */
  private double m_inconsistencyRateWhole;

  /** Inconsistency rate of the current selected featureset */
  private double m_inconsistencyRateCurrent;

  /** Inconsistency rate of the temp selected featureset */
  private double m_tmpInconsistencyRate;

  /** Inconsistency contribute of the features */
  private double [] m_inconsistencyContribute;

  /** Number of class value */
  private int m_numClassValue;

  /** Store time information */
  private long [] m_times;

  /** the count of invocation of equals function in hashkey */
  public static long m_fEqualsInvocationCount;

  /** the count of invocation of equals function in hashkey */
  public static long m_fCollisionCount;

  /** the count of invocation of hashCode function in hashkey */
  public static long m_fHashcodeCount;

  /** whether the class is invoked by weka explorer */
  public boolean m_fromWekaExplorer;

  /** the features ranked by their infoGain Contribution */
  int [] m_infoGainRankedFeatures;

  /**
   * Class providing keys to the hash table.
   */
  public class hashKey implements Serializable {

    /** Array of attribute values for an instance */
    private double [] attributes;

    /** True for an index if the corresponding attribute value is missing. */
    private boolean [] missing;

    /** The feautes will be included in the featureset.
     *  initially all features are included */
    private boolean [] included;

    /** The key */
    private int key;
    private int tmpKey;

    /**
     * Constructor for a hashKey
     *
     * @param t an instance from which to generate a key
     * @param numAtts the number of attributes
     */
    public hashKey(Instance t, int numAtts) throws Exception {

      int i;
      int cindex = t.classIndex();

      key = -999;
      attributes = new double [numAtts];
      missing = new boolean [numAtts];
      included = new boolean [numAtts];
      for (i=0;i<numAtts;i++) {
        included [i] = true;
        if (i == cindex) {
          missing[i] = true;
          included [i] = false;
        } else {
          if ((missing[i] = t.isMissing(i)) == false) {
            attributes[i] = t.value(i);
          }
        }
      }
    }

    /**
     * Constructor for a hashKey
     *
     * @param t an array of feature values
     */
    public hashKey(double [] t) {

      int i;
      int l = t.length;

      key = -999;
      attributes = new double [l];
      missing = new boolean [l];
      included = new boolean [l];
      for (i=0;i<l;i++) {
        included[i] = true;
        if (t[i] == Double.MAX_VALUE) {
          missing[i] = true;
        } else {
          missing[i] = false;
          attributes[i] = t[i];
        }
      }
    }

    /**
     * Return a hash code
     *
     * @return the hash code as an integer
     */
    public int hashCode() {

      INTERACT.m_fHashcodeCount++;
      if (key != -999)
        return key;
      else return getHashCode();

    }

    /**
     * Calculates a hash code
     *
     * @return the hash code as an integer
     */
    private int getHashCode() {
      int hv = 0;

      for (int i = 0; i < attributes.length; i++) {
        if (included[i]){
          if (missing[i]) {
           hv += (i * 13);
          }
          else {
            hv += (i * 5 * (attributes[i] + 1)); //the effecitivity of this hashing function
          }                                      //should be verified
        }
      }
      key = hv;

      return hv;
    }

    /**
     * Updata a hash code
     *
     * @param index the feature will be eliminate
     * from the feature set
     *
     * @return the hash code as an integer
     * if a feature has been eliminated before
     * the function will return -999
     */
    public int eliminateFeature(int index) {

      if (included[index]){
        included[index] = false;
      }
      else {
        return -999;
      }

      tmpKey = key;

      if (key != -999)
      {
        if (missing[index]) {
          key = key - index*13;
        }
        else {
          key -= (index*5*(attributes[index] + 1));
        }
        return key;
      }
      else {
        key = getHashCode();
        return key;
      }
    }

    /**
     *
     * @param int index, the feature should be
     * restored in the selected features set
     *
     */
    public void restoreFeature(int index){
      if (!included[index]){
        included[index] = true;
        key = tmpKey;
      }
      else{
        System.out.print("The feature hasn't been eliminate. Error!");
        System.exit(9);
      }
    }

    /**
     * Tests if two instances are equal
     *
     * @param b a key to compare with
     */
    public boolean equals(Object b) {

      INTERACT.m_fEqualsInvocationCount++;

      if ((b == null) || !(b.getClass().equals(this.getClass()))) {
    	  INTERACT.m_fCollisionCount++;
        return false;
      }
      boolean ok = true;
      boolean l, l1;
      if (b instanceof hashKey) {
        hashKey n = (hashKey)b;
        for (int i=0;i<attributes.length;i++) {
          l = n.missing[i];
          l1 = n.included[i];

          if(l1 != included[i]){
            ok = false;
            INTERACT.m_fCollisionCount++;
            break;
          }
          else if (l1){
            if (missing[i] || l) {
              if ((missing[i] && !l) || (!missing[i] && l)) {
                ok = false;
                INTERACT.m_fCollisionCount++;
                break;
              }
            } else {
              if (attributes[i] != n.attributes[i]) {
                ok = false;
                INTERACT.m_fCollisionCount++;
                break;
              }
            }
          }              //else if l1
        }                //for i
      } else {
    	  INTERACT.m_fCollisionCount++;
        return false;
      }
      return ok;
    }

    /**
     * Prints the hash code
     */
  public void print_hash_code() {

      System.out.println("Hash val: "+hashCode());
    }
  } //the end of the hashkey class

  /**
   * Constructor
   */

  public INTERACT() {
    super();
    m_fromWekaExplorer = true;
  }

  /**
   * Constructor.
   *
   * @param args the parameter got from commandline
   **/
  public INTERACT(String[] args) {
    super(args);
    m_fromWekaExplorer = false;

  }

  /**
   * Returns a string describing this search method
   * @return a description of the search suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "INTERACT : \n\nINTERACT algorithm is a feature selection method based "
      +"on Inconsistency and Symmetrical Uncertainty Measurement for "
      +"feature interaction analysis. " 
      +"Use in conjunction with attribute evaluators (Inconsistency, "
      +"SymmetricUncertainty)  Metric combination.\n";
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
   */
  public int[] search (ASEvaluation ASEval, Instances data)
    throws Exception {

    try{
    InitialData(data);
    }
    catch (Exception e) {
      System.out.print("\n Data Initial Error\n"
                   + e.toString() + "\n\n");
      System.exit(9);
    }

    double[][] tmpSelectedFeatures = rankedAttributes();
    int[] SelectedFeatures = new int[tmpSelectedFeatures.length];

    for (int i = 0; i < tmpSelectedFeatures.length; i++) {
      SelectedFeatures[i] = (int)tmpSelectedFeatures[i][0];
    }

    return  SelectedFeatures;
  }

  /**
   * Sorts the evaluated attribute list
   *
   * @return an array of sorted (highest eval to lowest) attribute indexes
   * @exception Exception of sorting can't be done.
   */
  public double[][] rankedAttributes ()
    throws Exception {
    int count = 0;

    INTERACTPack();

    for(int i=0;i<m_currentFeatureSet.length;i++)
    {
      if (m_currentFeatureSet[i] == true) {
        count++;
      }
    }

    double[][] result = new double [count][2];
    double[] forRank = new double [count];
    double[] rankTag = new double [count];

    //Rank features according their InfoGainContri

    count = 0;

    for(int i=0;i<m_currentFeatureSet.length;i++)
    {
      if (m_currentFeatureSet[i] == true) {
        rankTag[count] = i;
        result[count][0] = i;
        result[count][1] = m_inconsistencyContribute[i];
        forRank[count] = m_inconsistencyContribute[i];
        count ++;
      }
    }

    int[] ranked = Utils.sort(forRank);

  // reverse the order of the ranked indexes

    count = 0;

    for (int i = ranked.length - 1; i >= 0; i--) {
      result[count++][0] = rankTag[ranked[i]];
    }

    // convert the indexes to attribute indexes
    for (int i = 0; i < ranked.length; i++) {
      int temp = ((int)result[i][0]);
      result[i][1] = m_inconsistencyContribute[temp];
    }

    return  result;
  }

  private void INTERACTPack() {


    System.out.print("-----------------------------------------------");
    System.out.print("\nRank features by SymmetryUncert Metric...\n");
    m_times[0] = System.currentTimeMillis();
    evaluateInfoGainContri();
    m_times[1] = System.currentTimeMillis();
    System.out.print("\nRanking finished (" + (m_times[1] - m_times[0]) +
                     ")\n");

    System.out.print("-----------------------------------------------");
    System.out.print("\nInitial Hashtable...\n");

    m_times[2] = System.currentTimeMillis();
    fillHashTable();
    m_times[3] = System.currentTimeMillis();
    System.out.print("\nInitial Hashtable finished (" + (m_times[3] - m_times[2]) +
                     ")\n");
    System.out.print("-----------------------------------------------");

    System.out.print("\nStart Aggressive Search...\n\n");
    m_times[4] = System.currentTimeMillis();
    aggressiveSearch();
    m_times[5] = System.currentTimeMillis();
    System.out.print("\nAggressive Search finished(" + (m_times[5] - m_times[4]) +
                     ")\n");
    System.out.print("-----------------------------------------------\n\n");
    resultPrint();
    System.out.print("\n\nExit...");

  }

  private void InitialData (Instances inst) throws Exception {

    m_times = new long [20];
    this.m_fEqualsInvocationCount = 0;
    this.m_fHashcodeCount = 0;
    this.m_fCollisionCount = 0;

    if(m_threshold <=0) {
      m_threshold =0.0001;
    }

    //alan: here we set the last attribute as the class label
    m_trainInstances=inst;
    m_trainInstances.setClassIndex(m_trainInstances.numAttributes() -1);

    if (m_trainInstances.checkForStringAttributes()) {
      throw  new UnsupportedAttributeTypeException("Can't handle string attributes!");
    }

    /*System.out.print("\nRemoves all instances with a missing\n"
                     +"class value from the dataset....");*/
    //Removes all instances with a missing class value from the dataset.
    m_trainInstances.deleteWithMissingClass();

    m_classIndex = m_trainInstances.classIndex();
    if (m_classIndex < 0) {
      throw new Exception("Consistency subset evaluator requires a class "
                          + "attribute!");
    }

    if (m_trainInstances.classAttribute().isNumeric()) {
      throw new Exception("Consistency subset evaluator can't handle a "
                          +"numeric class attribute!");
    }
    System.out.print(" finished.\n");
    System.gc();

    m_numAttribs = m_trainInstances.numAttributes();
    m_numInstances = m_trainInstances.numInstances();

    m_times[0] = System.currentTimeMillis();
    System.out.print("\nDiscretize Dataset...");
    m_disTransform = new Discretize();
    m_disTransform.setUseBetterEncoding(true);
    m_disTransform.setInputFormat(m_trainInstances);
    m_trainInstances = Filter.useFilter(m_trainInstances, m_disTransform);
    System.out.print(" finished.");
    m_times[1] = System.currentTimeMillis();
    System.out.print(" (" + (m_times[1] - m_times[0]) +
                     ")\n");    

    System.gc();

    m_inconsistencyContribute = new double[m_numAttribs-1];
    m_currentFeatureSet = new boolean[m_numAttribs];
    m_mustIncludedFeatures = new boolean[m_numAttribs];

    m_startRange.setUpper(m_numAttribs - 1);
    m_starting = m_startRange.getSelection();

    for(int i=0;i<m_numAttribs;i++)
    {
      m_currentFeatureSet[i] =true;
      if (inStarting(i)) {
        m_mustIncludedFeatures[i] = true;
      }
      else {
        m_mustIncludedFeatures[i] = false;
      }
    }

    m_currentFeatureSet[m_classIndex]=false;

    m_numClassValue = m_trainInstances.classAttribute().numValues();

  }

  private void evaluateInfoGainContri()
  {
    m_attributeMerit = new double[m_numAttribs - 1];

    SymmetricalUncertAttributeEval eval = new SymmetricalUncertAttributeEval();

    try {
      eval.buildEvaluator(m_trainInstances);
    }
    catch (Exception e) {
      System.out.print("error ocurrs when build SymmetriUncert\n"+e.toString());
      System.exit(9);
    }


    for (int i = 0; i < m_numAttribs - 1; i++) {
      if (m_mustIncludedFeatures[i]) {
        m_attributeMerit[i] = Double.MAX_VALUE;
      }
      else {
        try {
          m_attributeMerit[i] = eval.evaluateAttribute(i);
        }
        catch (Exception e) {
          System.out.print("error occurs when evaluate feature "+ i +" using SymmetricalUncert.\n"+e.toString());
        }
      }
    }

    m_infoGainRankedFeatures = Utils.sort(m_attributeMerit);
  }

  private void fillHashTable(){
    System.out.print("\nAllocate Hashtable one...");
    m_table = new Hashtable((int)(m_numInstances * 1.5));
    System.out.print(" finished.\n");

    System.out.print("\nFill Hashtable one...");
    m_times[8] = System.currentTimeMillis();
    try{
      fillHashtableSubset(m_currentFeatureSet);
    }catch(Exception e){
      System.out.print(e+"\n fillHashtableSubset");
      System.exit(9);
    }
    m_inconsistencyRateWhole = InconsistencyCount(m_table);
    m_inconsistencyRateCurrent = m_inconsistencyRateWhole;
    m_times[9] = System.currentTimeMillis();
    if(!m_fromWekaExplorer){
      m_trainInstances = null;
    }
    System.gc();
    System.out.print(" finished. ("+(m_times[9]-m_times[8])+")\n");

    System.out.print("\nAllocate Hashtable two...");
    m_tmpTable = new Hashtable((int)(m_numInstances * 1.5));
    System.out.print(" finished.\n");
  }

  /**
   * Fill the Hash Table
   * @return double
   */
  private void fillHashtableSubset(boolean[] subset) throws Exception {
    //alan: A subset should not contain the class

    int [] fs;   //alan: which attribute is included in the subset
    int i;
    int count = 0;

    for (i=0;i<m_numAttribs;i++) {
      if (subset[i] == true) {
        count++;
      }
    }

    double [] instArray = new double[count];
    int index = 0;
    fs = new int[count];
    for (i=0;i<m_numAttribs;i++) {
      if (subset[i] == true) {
        fs[index++] = i;
      }
    }

    for (i=0;i<m_numInstances;i++) {
      Instance inst = m_trainInstances.instance(i);
      for (int j=0;j<fs.length;j++) {
        if (fs[j] == m_classIndex) {
          throw new Exception("A subset should not contain the class!");
        }
        if (inst.isMissing(fs[j])) {
          instArray[j] = Double.MAX_VALUE;
        } else {
          instArray[j] = inst.value(fs[j]);
        }
      } // for j
      insertIntoTable(inst, instArray);
    } //for i
  }

  /**
   * Inserts an instance into the hash table
   *
   * @param inst instance to be inserted
   * @param instA the instance to be inserted as an array of attribute
   * values.
   * @exception Exception if the instance can't be inserted
   */
  private void insertIntoTable(Instance inst, double [] instA)
       throws Exception {

    double [] tempClassDist2;
    double [] newDist;
    hashKey thekey;

    thekey = new hashKey(instA);
    //hashkey monitor
    //System.out.print(thekey.hashCode()+"\n");

    // see if this one is already in the table
    tempClassDist2 = (double []) m_table.get(thekey);
    if (tempClassDist2 == null) {
      newDist = new double [m_trainInstances.classAttribute().numValues()];
      newDist[(int)inst.classValue()] = inst.weight();

      // add to the table
      m_table.put(thekey, newDist);
    } else {
      // update the distribution for this instance
      tempClassDist2[(int)inst.classValue()]+=inst.weight();

      // update the table
      m_table.put(thekey, tempClassDist2);
    }
  }

  /**
   * calculates the level of inconsistency in a dataset using a subset of
   * features. The inconsistency of a hash table entry is the total number
   * of instances hashed to that location minus the number of instances in
   * the largest class hashed to that location. The total inconsistency is
   * the sum of the individual consistencies divided by the
   * total number of instances.
   *
   * @param table The hashtable based on which the inconsistency is calculated.
   *
   * @return the inconsistency of the hash table as a value between 0 and 1.
   */
  private double InconsistencyCount(Hashtable table) {
    Enumeration e = table.keys();
    double [] classDist;
    double count = 0.0;

    while (e.hasMoreElements()) {
      hashKey tt = (hashKey)e.nextElement();
      classDist = (double []) table.get(tt);
      count += Utils.sum(classDist);              //the value distribution in the class.
      int max = Utils.maxIndex(classDist);
      count -= classDist[max];
    }

    count /= (double)m_numInstances;
    return count;
  }

  /**
   * Search for features to be selected.
   */
  private void aggressiveSearch(){

    double InconsistencyContribution;
    int theFeature;

    for(int i=0;i<m_numAttribs-1;i++)
    {
      theFeature = m_infoGainRankedFeatures[i];

      if (m_mustIncludedFeatures[theFeature]) {
        continue;
      }

      //System.out.print("F"+String.valueOf(i+1)+", ");
      InconsistencyContribution = InconContri(theFeature);
      m_inconsistencyContribute[theFeature] = InconsistencyContribution;
      if (InconsistencyContribution<m_threshold) {
        removeFeature(theFeature);
      } else {
        restoreHashkeys(theFeature);
      }
      /*System.out.print("Finished:"+Utils.doubleToString(InconsistencyContribution,5,4)+"| ");
      if (i%5==0 && i!=0){
        System.out.print("\n");
      }*/
    }
  }

  /**
   * Get the inconsistency contribution of a feature
   * @param index int the index of the feature which will be evaluated
   * @return double the inconsistency contribution of the feature
   */
  private double InconContri(int index)
  {
    fillTmpHashtable(index);
    m_tmpInconsistencyRate = InconsistencyCount(m_tmpTable);
    return m_tmpInconsistencyRate - m_inconsistencyRateCurrent;
  }

  private void fillTmpHashtable (int index)
  {
    double [] tempClassDist2;
    double [] classDist;
    hashKey thekey;

    Enumeration e = m_table.keys();

    while (e.hasMoreElements()) {
      thekey = (hashKey)e.nextElement();
      classDist = (double []) m_table.get(thekey);
      thekey.eliminateFeature(index);

      // see if this one is already in the tmptable
      tempClassDist2 = (double[]) m_tmpTable.get(thekey);

      if (tempClassDist2 == null) {
        // add to the table
        tempClassDist2 = new double [classDist.length];
        for(int i=0;i<classDist.length;i++){
          tempClassDist2 [i] = classDist [i];
        }
        m_tmpTable.put(thekey, tempClassDist2);
      }
      else {
        // update the distribution for this instance
        for(int i=0;i<classDist.length;i++){
          tempClassDist2[i] += classDist[i];
        }
        // update the table
        m_tmpTable.put(thekey, tempClassDist2);
      }
    }
  }

  /**
   * remove feature i from the feature set
   * @param index int the index of the feature
   * which will been removed from the feature set.
   */
  private void removeFeature (int index) {

    m_inconsistencyContribute[index] = m_tmpInconsistencyRate
                                 -m_inconsistencyRateCurrent;

    m_inconsistencyRateCurrent = m_tmpInconsistencyRate;
    Hashtable tmp = m_table;
    m_table = m_tmpTable;
    m_tmpTable = tmp;
    m_tmpTable.clear();
    m_currentFeatureSet[index] = false;
  }

  /**
   * restore the hashkeys in the hashtable and clear the
   * m_tmptable.
   * @param index int the feature should the re-included
   * in the selected feature.
   */
  private void restoreHashkeys(int index) {

    hashKey thekey;

    Enumeration e = m_table.keys();

    while (e.hasMoreElements()) {
      thekey = (hashKey)e.nextElement();
      thekey.restoreFeature(index);
    }
    m_tmpTable.clear();

  }

  public String toString () {
    StringBuffer BfString = new StringBuffer();
    BfString.append("\nINTEACT Feature Selection.\n");

    if (m_starting != null) {
      BfString.append("\tAlways include attributes: ");

      BfString.append(startSetToString());
      BfString.append("\n");
    }

    if (m_threshold != -Double.MAX_VALUE) {
      BfString.append("\tThreshold for discarding attributes: "
                      + Utils.doubleToString(m_threshold,8,4)+"\n");
    }

    return BfString.toString();
  }

  public void resultPrint () {

    int count =0;
    int printcounter = 0;

    StringBuffer BfString = new StringBuffer();

    BfString.append("DataSet name: "+m_fileName+"\n");

    BfString.append("Total featurs: "+(this.m_numAttribs-1)+
                     "\nTotal Instances: "+this.m_numInstances+
                     "\nTotal Class Lables: "+m_numClassValue+"\n");

    BfString.append("The inconsistency rate of the whole dataset is: "
                      +String.valueOf(this.m_inconsistencyRateWhole)+"\n");

    BfString.append("The inconsistency rate of the selected features set is: "
                      +String.valueOf(this.m_inconsistencyRateCurrent)+"\n");

    BfString.append("The increase ratio is : "
                      +String.valueOf(m_inconsistencyRateCurrent/m_inconsistencyRateWhole)+"\n");

    BfString.append("The feature selection threshold is: "
                      +String.valueOf(this.m_threshold)+"\n");

    BfString.append("Entries in hashtable: "
                      +String.valueOf(this.m_table.size())+"\n");

    for(int i=0;i<m_currentFeatureSet.length;i++)
    {
      if (m_currentFeatureSet[i] == true) {
        count ++;
      }
    }

    BfString.append("Selected "+String.valueOf(count)+" features from "+
                     String.valueOf(this.m_numAttribs-1)+" features.\n");
    BfString.append("Selected ratio: "+ String.valueOf((float)count/((float)m_numAttribs-1))+"\n");

    BfString.append("The Selected Features are: \n");

    printcounter = 0;
    for(int i=0;i<m_currentFeatureSet.length;i++) {
      if (m_currentFeatureSet[i] == true) {
        printcounter ++;

        BfString.append("F"+String.valueOf(i+1)+" : InCon:"+Utils.doubleToString(m_inconsistencyContribute[i],5,4)+", SymU:" +Utils.doubleToString(m_attributeMerit[i],5,4)+"; ");
        if (printcounter % 5 == 0) {
          BfString.append("\n");
        }
      }
    }

    printcounter = 0;
    BfString.append("\nThe Unselected Features are: \n");
    for(int i=0;i<m_currentFeatureSet.length;i++){
      if(m_currentFeatureSet[i]!=true && i!=m_classIndex)
      {
        printcounter++;
        BfString.append("F"+String.valueOf(i+1)+" : InCon:"+Utils.doubleToString(m_inconsistencyContribute[i],5,4)+" SymU:" +Utils.doubleToString(m_attributeMerit[i],5,4)+"; ");
        if (printcounter%5 == 0 && i!=0) {
          BfString.append("\n");
        }
      }
    }

    System.out.print(BfString.toString());
  }

  public static void main(String[] args) {
    INTERACT interact = new INTERACT(args);
    try{
    	interact.search(null, interact.m_trainInstances);
    }
    catch(Exception e)
    {
      System.out.print(e);
    }
    System.exit(0);
  }

}