import java.util.*;
import java.text.*;
import java.io.*;
import pack1.*;

class Main {
  public static final int N_MAX = 5007;
  public static int W,L;
  public static ToaDo BASE;
  public static int carNum,periodNum,toaDoNum;
  public static double banKinh;
  public static ToaDoMeThod mt ;
  public static ArrayList<ToaDo> tapSensor ;
  public static ArrayList<ToaDo> tapRelay ;
  public static ArrayList<Edge> tapEdge ;
  public static ArrayList<Edge> tapSpanning ;
  public static ArrayList<Edge> tapSteiner;
  public static int[] root;
  public static DanhSachDinhKe[] adj;
  public static boolean[] visited = new boolean[N_MAX];
  public static int[] truyVet = new int[N_MAX];
  public Main() {
	  mt  = new ToaDoMeThod();
	  tapSensor = new ArrayList<ToaDo>();
	  tapRelay = new ArrayList<ToaDo>();
	  tapEdge = new ArrayList<Edge>();
	  tapSpanning = new ArrayList<Edge>();
	  tapSteiner  = new ArrayList<Edge>();
	  
  }
  public static void print(double s){
    System.out.println(s);
  }
  public static void print(String s){
    System.out.println(s);
  }

  public void docFile(String filein) throws IOException{
    //File f1 = new File("test1.txt");
    Scanner sc = new Scanner(new File(filein));
    W = sc.nextInt();
    L = sc.nextInt();
    BASE = new ToaDo(sc.nextDouble(),sc.nextDouble());
    carNum = sc.nextInt();
    banKinh = sc.nextDouble();
    periodNum = sc.nextInt();
    toaDoNum = carNum * periodNum;

    for (int i=0;i < carNum;i++)
      for (int j=0;j < periodNum;j++)
        tapSensor.add(new ToaDo(sc.nextDouble(),sc.nextDouble(),j));
    //-----------------------
    tapSensor.add(BASE);
    toaDoNum ++;
  }
  public static void sortEdge(){
    Collections.sort(tapEdge, new Comparator<Edge>() {
      public int compare(Edge e1,Edge e2){
        if (e1.doDai > e2.doDai)
          return 1;
        else {
          if (e1.doDai == e2.doDai)
            return 0;
            else return -1;
        }
      }
    });
  }
  public static int getRoot(int x){
    if (root[x] == x) return x;
    else
      return( root[x] = getRoot(root[x]) );
  }
  public static void addRelay(ToaDo d1,ToaDo d2){
    // them cac relay node noi giua d1 va d2
    double R = banKinh;
    if (mt.khoangCach(d1,d2) <= 2 * R) return;
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
    tapRelay.add(d3);
    addRelay(d3,d2);
  }
  public void spanningTree() throws IOException{

    for (int i=0;i < tapSensor.size();i++)
    for (int j=i+1;j < tapSensor.size();j++){
      ToaDo p1 = tapSensor.get(i);
      ToaDo p2 = tapSensor.get(j);
      tapEdge.add(new Edge(i,j,p1,p2,mt.khoangCach(p1,p2)));
    }
    //----------------------------
    sortEdge();
    //---------------------------
    ToaDo p1,p2;
    //chuan bi cho spanning tree
    root = new int[toaDoNum];
    for (int i=0;i < toaDoNum;i++) root[i] = i;

    for (Edge e: tapEdge){
      int idA = e.id1;
      int idB = e.id2;
      int p = getRoot(idA);
      int q = getRoot(idB);
      if (p == q) continue;
      root[p] = q;
      tapSpanning.add(e);
      //addRelay(e.point1,e.point2);
    }
  }
  public void steinerTree(){
    int idCanh,d1=0,d2=0,d3=0;
    int id1=0,id2=0,id3=0,idSteiner=0;
    ToaDo d1Choosed= new ToaDo(0,0);
    ToaDo d2Choosed= new ToaDo(0,0);
    ToaDo d3Choosed= new ToaDo(0,0);
    ToaDo steinerP;
    Edge e1 = new Edge(0,0,d1Choosed,d1Choosed,0);
    Edge e2 = new Edge(0,0,d1Choosed,d1Choosed,0);
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
        if (e1.id1 == e2.id1){
          d1 = e1.id2; d3 = e2.id2; d2 = e1.id1; giaoNhau = true;
        }
        //------------------------------------------
        if (e1.id2 == e2.id2){
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

      if (mt.lessThan120(d1Choosed,d2Choosed,d3Choosed)){
        steinerP = mt.diemSteiner(d1Choosed,d2Choosed,d3Choosed);
        //----------------------------------------------
        
        double kc1 = mt.khoangCach(steinerP,d1Choosed);
        double kc2 = mt.khoangCach(steinerP,d2Choosed);
        double kc3 = mt.khoangCach(steinerP,d3Choosed);
        if ((kc1 < 2*banKinh) || (kc2 < 2*banKinh) || (kc3 < 2*banKinh)) continue;
        
        //---------------------------------------------
        e1.isXoa = true;
        e2.isXoa = true;
        tapSensor.add(steinerP);
        //---------------------
        /*
        steinerP.print();
        d1Choosed.print();
        d2Choosed.print();
        d3Choosed.print();
        print("");
        */
        //------------------------
        idSteiner = tapSensor.size() - 1;

        tapSpanning.add(new Edge(id1,idSteiner,d1Choosed,steinerP,0) );
        tapSpanning.add(new Edge(id2,idSteiner,d2Choosed,steinerP,0) );
        tapSpanning.add(new Edge(id3,idSteiner,d3Choosed,steinerP,0) );

        tapRelay.add(steinerP);
      }
    }

    for (Edge e: tapSpanning)
      if (!e.isXoa) 
        tapSteiner.add(e);
    
    // chuan bi cho danh sach dinh ke
    adj = new DanhSachDinhKe[tapSensor.size()];
    for (int i=0;i < tapSensor.size();i++) 
      adj[i] = new DanhSachDinhKe();
    //----------------------------------
    for (Edge e: tapSteiner){
      addRelay(e.point1,e.point2);
      adj[e.id1].push(e.id2);
      adj[e.id2].push(e.id1);
    }
  }
  public void ghiFile(String fileout) throws IOException{

    File f2 = new File(fileout);
    FileWriter fw = new FileWriter(f2);

    DecimalFormat df = new DecimalFormat("0.000");
    for (ToaDo p: tapRelay){
      fw.write("(x-" + df.format(p.x) +")^2 +(y-" + df.format(p.y) + ")^2 = " + df.format(banKinh * banKinh)) ;
      fw.write("\n");
    }

    for (int i=0;i < toaDoNum;i++)
      fw.write(tapSensor.get(i).toDecimalString() + "\n");

    fw.close();

    print("ADDED = "+tapRelay.size());
  }
  public static void datRelay(int u){
    //neu u la Steiner point hoac da duoc dat relay thi back
    if (tapSensor.get(u).chuKy == -1) return;
    //-------------------------------------
    tapRelay.add(tapSensor.get(u));
    tapSensor.get(u).chuKy = -1;
  }
  public static void truyVetBack(int u){
    int v = u;
    int ck = tapSensor.get(u).chuKy;
    // neu u la 1 Steiner point hoac BASE thi thoat
    if (ck == -1) return;
    //----------------------------
    while (v != toaDoNum - 1){
      v = truyVet[v];
      if (tapSensor.get(v).chuKy != ck)
        datRelay(v);
    }
  }
  public static void dfs(int u){
    if (visited[u]) return;
    visited[u] = true;
    truyVetBack(u);

    for (int i=0;i < adj[u].size();i++){
      int v = adj[u].get(i);
      if (!visited[v]){
        truyVet[v] = u;
        dfs(v);
      }
    }
  }
  public void dfsMethod(){
    truyVet[toaDoNum - 1] = -1;
    for (int i=0;i < tapSensor.size();i++) 
      visited[i] = false;
    dfs(toaDoNum - 1);
  }
  public static void main(String[] args) throws IOException{
    //------------------------------------
		 for(int i = 1 ; i<= 18; i++) {
			 System.out.println("Test-Case: " +i);
			 long Starttime = System.nanoTime();
			 Main x = new Main();
			 String path = "./Testmip/Test";
			 x.docFile(path+i+".inp");
			 x.spanningTree();
			 x.steinerTree();
			 x.dfsMethod();
			 x.ghiFile(path+i+".out");
			 long End = System.nanoTime();
			 double time = (End -Starttime)*1E-9;
			 System.out.println("time = " + time);
			  System.out.println("--------------------------");
		 }

  }
}