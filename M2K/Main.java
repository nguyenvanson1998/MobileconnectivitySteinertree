import java.util.*;
import java.text.*;
import java.io.*;
import pack1.*;

class Main {
  public static ToaDo base;
  public static int width, length, carCount, periodCount;
  public static CanhNoi[] tapCanh;
  public static int tapCanhCount = 0;
  public static int[] root;
  public static final double oo = (double) 1e9;
  public static double R;
  public static final int N_MAX = 2000;
  public static final int INVALID = -1;
  public static int n,k;
  public static ToaDo[] tapDiem; // tapDiem[i] = diem thu i(0 <= i <= n-1)
  public static Cluster[] tapCluster;// tapCluster[i] = cluster thu i (0 <= i <= k-1)
  public static int[] newCluster = new int[N_MAX];
  public static int[] oldCluster = new int[N_MAX];
  public static List<ToaDo> addedPoint;
  public static ArrayList<CanhNoi> tapSpanning;
  public static ArrayList<ToaDo> tapSensor ;
  public static SteinerCalculator mt ;

  public Main() {
	  addedPoint = new ArrayList<ToaDo>();
	  tapSpanning = new ArrayList<CanhNoi>();
	  tapSensor = new ArrayList<ToaDo>();
	  mt =  new SteinerCalculator();
  }
  public void nhapDuLieu(String filein) throws Exception{
    Scanner sc = new Scanner(new File(filein));
    //----------------------
    //System.out.println("Nhap width,length: ");
    width = sc.nextInt();
    length = sc.nextInt();
    //System.out.println("Nhap toa do base station: ");
    base = new ToaDo(sc.nextDouble(),sc.nextDouble());
    //---------------------
    //System.out.println("Nhap so xe: ");
    carCount = sc.nextInt();
    //System.out.println("Nhap ban kinh truyen tin: ");
    R = sc.nextDouble();
    //System.out.println("Nhap so period: ");
    periodCount = sc.nextInt();
    //---------------------
    n = carCount * periodCount;
    //System.out.println("Nhap k: ");
    //k = sc.nextInt();
    k = periodCount * 16 / ((int) R * (int) R);
    if(k > n) {
    	k = n/2;
    }
    //k = 30;
    //---------------------
    tapDiem = new ToaDo[n];
    //System.out.println("Nhap n diem: ");
    for (int i=0;i < n;i++)
      tapDiem[i] = new ToaDo(sc.nextDouble(),sc.nextDouble());
  }
  public void inDuLieu(String fileout) throws IOException{
    //in cac mobile sensor va base
    File f2 = new File(fileout);
    FileWriter fw = new FileWriter(f2);
    //------------------------------------
    for (ToaDo diem: tapDiem)
      fw.write(diem.toDecimalString() + "\n");
    fw.write(base.toDecimalString() + "\n");
    fw.write("\n");
    //-----------------------------------
    // in cac static sensor duoc them vao
    for (ToaDo diem: addedPoint){
      fw.write(diem.toCircle(R) + "\n");
    }
    fw.close();
    System.out.println(addedPoint.size() + " points added");
  }
  public void randomCum(){
    tapCluster = new Cluster[k];
    Random rd = new Random();
    int numMax = n-1,numMin = 0,numRan;
    boolean[] duocChon = new boolean[n];
    for (int i=0;i < n;i++) 
      duocChon[i] = false;

    for (int i=0;i < k;i++){
      // chon ngau nhien not diem trong n diem lam centroid cua Cluster i
      while (true){
        numRan = rd.nextInt(numMax - numMin + 1) + numMin;
        if (!duocChon[numRan]) break;
      }
      duocChon[numRan] = true;
      // dat centroid cho Cluster i
      tapCluster[i] = new Cluster(tapDiem[numRan]);
    }
  }
  public static boolean newSameOld(){
    //= true neu thanh vien cua cac Cluster khong thay doi
    for (int i=0;i < n;i++)
      if (oldCluster[i] != newCluster[i]) return false;
    return true;
  }
  public void phanCum(){

    boolean done = false;
    double kCachMin;
    int indexOfCluster;
    for (int i=0;i < n;i++) oldCluster[i] = INVALID;

    while (!done){
      
      for (int i=0;i < n;i++){
        // tim Cluster gan nhat voi diem i
        kCachMin = tapDiem[i].khoangCach(tapCluster[0].getCentroid());
        indexOfCluster = 0;

        for (int j=1;j < k;j++){
          double kCach = tapDiem[i].khoangCach(tapCluster[j].getCentroid());
          if (kCach < kCachMin){
            kCachMin = kCach;
            indexOfCluster = j;
          }
        }
        // them diem i vao Cluster gan nhat
        tapCluster[indexOfCluster].add(tapDiem[i]);
        newCluster[i] = indexOfCluster;
      }
      // kiem tra dieu kien dung
      if (newSameOld()) {done = true;break;} 
      // chuyen New thanh Old 
      for (int i=0;i < n;i++) 
        oldCluster[i] = newCluster[i];
      // dat lai cac centroid
      for (int i=0;i < k;i++){
        tapCluster[i] = new Cluster(tapCluster[i].center());
      }
    }
  }
  public static void addPoint(ToaDo d1,ToaDo d2){
    // them cac sensor noi giua d1 va d2
    if (d1.khoangCach(d2) <= 2 * R) return;
    ToaDo d3 = new ToaDo(0,0);
    double kCach = d1.khoangCach(d2);
    double deltaX = 2*R* Math.abs(d2.getX() - d1.getX())/kCach;
    double deltaY = 2*R* Math.abs(d2.getY() - d1.getY())/kCach;
    //-----------------------
    if (d1.getX() < d2.getX())
      d3.setX(d1.getX() + deltaX);
    else
      d3.setX(d1.getX() - deltaX);
    //------------------------
    if (d1.getY() < d2.getY())
      d3.setY(d1.getY() + deltaY);
    else
      d3.setY(d1.getY() - deltaY);
    //------------------------------
    addedPoint.add(d3);
    addPoint(d3,d2);
  }
  public void addPointOfCluster(){
    List<ToaDo> pointOfThisCluster; // cac diem trong 1 cluster nao do
    double[] toAddedPointMin; // khoang cach min tu 1 diem thuoc cluster den 1 diem trong addedPoint
    int[] idAddedPoint; // index cua diem do trong addedPoint
    int idBegin; // index bat dau trong addedPoint
    int pointCount; // so luong diem thuoc 1 cluster 
    int connectedCount; // so luong diem thuoc 1 cluster da duoc ket noi
    boolean[] connected; // diem i da duoc ket noi chua
    ToaDo centroid,pointMin,plusPoint,point;

    for (int clus = 0;clus < k;clus ++){
      pointOfThisCluster = tapCluster[clus].returnList();
      pointCount = pointOfThisCluster.size();
      centroid = tapCluster[clus].getCentroid();
      addedPoint.add(centroid);
      toAddedPointMin = new double[pointCount];
      connected = new boolean[pointCount];
      idAddedPoint = new int[pointCount];
      connectedCount = 0;

      for (int i=0;i < pointCount;i++){
        toAddedPointMin[i] = pointOfThisCluster.get(i).khoangCach(centroid);
        idAddedPoint[i] = addedPoint.size() - 1;
        connected[i] = false;
      }

      while (connectedCount < pointCount){
        
        double kcMin = oo;
        int idMin = -1;
        for (int i=0;i < pointCount;i++)
          if ((!connected[i]) && (toAddedPointMin[i] < kcMin)){
            kcMin = toAddedPointMin[i];
            idMin = i;
          }
        if (idMin == -1) {
          System.out.println("Error");break;
        }
        //------------------------------------
        connected[idMin] = true;
        connectedCount ++;
        pointMin = pointOfThisCluster.get(idMin);
        idBegin = addedPoint.size();
        //luu lai idBegin truoc khi addPoint
        plusPoint = addedPoint.get(idAddedPoint[idMin]);
        addPoint(pointMin,plusPoint);
        //------------------------------------
        for (int i=0;i < pointCount;i++)
        if (!connected[i])
          for (int j = idBegin;j < addedPoint.size();j++){
            point = pointOfThisCluster.get(i);
            plusPoint = addedPoint.get(j);
            if (point.khoangCach(plusPoint) < toAddedPointMin[i]){
              toAddedPointMin[i] = point.khoangCach(plusPoint);
              idAddedPoint[i] = j;
            }
          }
      }
    }
  }
  public static void sortTapCanh(){
    CanhNoi tmp;
    for (int i=0;i < tapCanhCount;i++)
    for (int j=i+1;j < tapCanhCount;j++)
      if (tapCanh[i].getLength() > tapCanh[j].getLength()){
        tmp = tapCanh[i];
        tapCanh[i] = tapCanh[j];
        tapCanh[j] = tmp;
      }
  }
  public static int getRoot(int x){
    if (root[x] == x) return x;
    else
      return( root[x] = getRoot(root[x]) );
  }
  public static void buildCayKhung(){
    ToaDo p1,p2;
    root = new int[k+1];
    for (int i=0;i <= k;i++) root[i] = i;

    for (int i=0;i < tapCanhCount;i++){
      int idA = tapCanh[i].id1;
      int idB = tapCanh[i].id2;
      int p = getRoot(idA);
      int q = getRoot(idB);
      if (p == q) continue;
      root[p] = q;
      //------------------------------------------
      if (idA < k) p1 = tapCluster[idA].getCentroid();
      else
        p1 = base;
      //------------------------------------------
      if (idB < k) p2 = tapCluster[idB].getCentroid();
      else
        p2 = base;
      //-----------------------------------------
      //addPoint(p1,p2);
      tapSpanning.add(new CanhNoi(idA,idB,0));
    }
  }
  public static void steinerTree(){
    //---------------------------------------------
    for (int i=0;i < k;i++)
      tapSensor.add(tapCluster[i].getCentroid());
    tapSensor.add(base);
    //---------------------------------------------

    int idCanh,d1=0,d2=0,d3=0;
    int id1=0,id2=0,id3=0,idSteiner=0;
    ToaDo d1Choosed= new ToaDo(0,0);
    ToaDo d2Choosed= new ToaDo(0,0);
    ToaDo d3Choosed= new ToaDo(0,0);
    ToaDo steinerP;
    CanhNoi e1 = new CanhNoi(0,0,0);
    CanhNoi e2 = new CanhNoi(0,0,0);
    double alpha,alphaMax;

    for (int i=0;i < tapSpanning.size();i++){
      if (tapSpanning.get(i).isXoa) 
        continue;
      idCanh = -1;
      alphaMax = - 10e9;
      for (int j=0;j < tapSpanning.size();j++){
        if (i == j) continue;
        if (tapSpanning.get(j).isXoa) continue;
        boolean giaoNhau = false;
        e1 = tapSpanning.get(i);
        e2 = tapSpanning.get(j);
        //------------------------------------------
        if (e1.id1 == e2.id1) {
          d1 = e1.id2; d3 = e2.id2; d2 = e1.id1; giaoNhau = true;
        }
        //------------------------------------------
        if (e1.id2 == e2.id2) {
          d1 = e1.id1; d3 = e2.id1; d2 = e1.id2; giaoNhau = true;
        }
        //------------------------------------------
        if (e1.id1 == e2.id2){
          d1 = e1.id2; d3 = e2.id1; d2 = e1.id1; giaoNhau = true;
        }
        //------------------------------------------
        if (e1.id2 == e2.id1){
          d1 = e1.id1; d3 = e2.id2; d2 = e1.id2; giaoNhau = true;
        }
        if (!giaoNhau) continue;
        alpha = mt.cosGocGiua(tapSensor.get(d1),tapSensor.get(d2),tapSensor.get(d3));

        if (alpha > alphaMax){
          alphaMax = alpha;
          idCanh = j;
          d1Choosed = tapSensor.get(d1); 
          d2Choosed = tapSensor.get(d2); 
          d3Choosed = tapSensor.get(d3);
          id1 = d1; id2 = d2; id3 = d3;
        }        
      }
      if (idCanh == -1) continue;
      e2 = tapSpanning.get(idCanh);

//      if (mt.lessThan120(d1Choosed,d2Choosed,d3Choosed)){
//        steinerP = mt.diemSteiner(d1Choosed,d2Choosed,d3Choosed);
//        //----------------------------------------------
//        
//        double kc1 = mt.khoangCach(steinerP,d1Choosed);
//        double kc2 = mt.khoangCach(steinerP,d2Choosed);
//        double kc3 = mt.khoangCach(steinerP,d3Choosed);
//        if ((kc1 < 2*R) || (kc2 < 2*R) || (kc3 < 2*R)) continue;
//        
//        //---------------------------------------------
//        e1.isXoa = true;
//        e2.isXoa = true;
//        tapSensor.add(steinerP);
//        //---------------------
//        //------------------------
//        idSteiner = tapSensor.size() - 1;
//
//        tapSpanning.add(new CanhNoi(id1,idSteiner,0) );
//        tapSpanning.add(new CanhNoi(id2,idSteiner,0) );
//        tapSpanning.add(new CanhNoi(id3,idSteiner,0) );
//
//        addedPoint.add(steinerP);
//      }
    }

    for (CanhNoi e: tapSpanning)
      if (!e.isXoa) 
        addPoint(tapSensor.get(e.id1),tapSensor.get(e.id2));
  }
  public void connectCluster(){
    ToaDo p1,p2;
    tapCanh = new CanhNoi[(k+1)*k/2];
    tapCanhCount = 0;

    for (int i=0;i <= k;i ++)
    for (int j=i+1;j < k+1;j++) {
      if (i < k) p1 = tapCluster[i].getCentroid();
      else
        p1 = base;
      //----------------------------------------
      if (j < k) p2 = tapCluster[j].getCentroid();
      else
        p2 = base;
      //----------------------------------------
      tapCanh[tapCanhCount++] = new CanhNoi(i,j,p1.khoangCach(p2));
    }

    sortTapCanh();
    buildCayKhung();
    steinerTree();
  }
  public static void main(String[] args) throws Exception{
	 for(int i = 1 ; i<= 18; i++) {
		 System.out.println("Test-Case: " +i);
	Main x = new Main();	 
	long Starttime = System.nanoTime();
	String path = "./Testmip/Test";
    x.nhapDuLieu(path+i+".inp");
    x.randomCum();
    x.phanCum();
    x.addPointOfCluster();
    x.connectCluster();
    x.inDuLieu(path+i+".out");
    long End = System.nanoTime();
    double time = (End -Starttime)*1E-9;
    System.out.println("time = " + time);
    System.out.println("--------------------------");
	 }
  }
}