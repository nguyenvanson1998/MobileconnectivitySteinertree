package test2;

import java.util.*;
import java.text.*;
import java.io.*;

import pack1.*;

class Main {
	public static File f2;
	public static FileWriter fw;
	public static double radius;
	public static final int N_MAX = 5000;
	public static final int oo = (int) 1e9;
	public static ToaDo[] tapDiem = new ToaDo[N_MAX];
	public static List<Sensor> tapGiao1;
	public static boolean[] covered =new boolean[N_MAX];
	public static int n;
	public static int numIter;
	// --------------------------------------------------------
	// i added something
	public static List<ToaDo> tapSensor;
	public static List<ToaDo> tapRelay ;
	public static List<Edge> tapEdge ;
	public static int[] root;
	public static int relayNum = 0;
	public static boolean[][] mark = new boolean[N_MAX][N_MAX];
	public static int W, L;
	public static Adjacent[] tapAdj;

	// ---------------------------------------------------------

	public Main() {
		tapGiao1  = new ArrayList<Sensor>();
		tapSensor   = new ArrayList<ToaDo>();
		tapRelay = new ArrayList<ToaDo>();
		
		tapEdge  = new ArrayList<Edge>();
		
	}
	public static ToaDo giaoDiem(ToaDo p1, ToaDo p2, double banKinh, int id) {
		/*
		 * Output: toa do giao diem cua 2 duong tron co cung ban kinh, co tam
		 * tai p1 va p2
		 */
		double x1 = p1.getX(), x2 = p2.getX();
		double y1 = p1.getY(), y2 = p2.getY();
		double xCenter = (x1 + x2) / 2;
		double yCenter = (y1 + y2) / 2;
		double distance = p1.khoangCach(p2);
		double xRes, yRes;
		if (y1 != y2) {
			double heSo = (x1 - x2) / (y2 - y1);
			xRes = xCenter
					+ Math.sqrt((banKinh * banKinh - distance * distance / 4.0)
							/ (1 + heSo * heSo));
			// -----------------------------------------
			yRes = heSo * (xRes - xCenter) + yCenter;
		} else {
			xRes = xCenter;
			yRes = yCenter
					+ Math.sqrt(banKinh * banKinh - distance * distance / 4.0);
		}
		// ----------------------------------------
		if (id == 2) {
			xRes = 2 * xCenter - xRes;
			yRes = 2 * yCenter - yRes;
		}
		return new ToaDo(xRes, yRes);
	}

	public void nhapDuLieu(String filein, String fileout) throws IOException {

		f2 = new File(fileout);
		fw = new FileWriter(f2);

		File fi = new File(filein);
		Scanner sc = new Scanner(fi);
		// ---------------------------------------------------
		System.out.println("Nhap");
		W = sc.nextInt();
		L = sc.nextInt();
		for (int i = 0; i <= 3000; i++)
			for (int j = 0; j <= 3000; j++)
				mark[i][j] = false;

		ToaDo BASE = new ToaDo(sc.nextDouble(), sc.nextDouble());
		int carNum = sc.nextInt();
		radius = sc.nextDouble() * 2;
		int periodNum = sc.nextInt();
		n = carNum * periodNum;
		// ---------------------------------------------------
		for (int i = 0; i < n; i++)
			tapDiem[i] = new ToaDo(sc.nextDouble(), sc.nextDouble());
		tapDiem[n++] = BASE;

		// ----------------------------------------
		// loc ra cac diem trung nhapDuLieu
		int nFake = 0;
		ToaDo[] tapDiemFake = new ToaDo[N_MAX];

		for (int i = 0; i < n; i++) {
			boolean isHave = false;
			for (int j = 0; j < i; j++)
				if (tapDiem[i].equal(tapDiem[j])) {
					isHave = true;
					break;
				}
			if (isHave)
				continue;
			tapDiemFake[nFake++] = tapDiem[i];
		}
		n = nFake;
		for (int i = 0; i < n; i++)
			tapDiem[i] = tapDiemFake[i];
		// ----------------------------------------
		// ----------------------------------------
	}

	public static boolean check(int x, int y, int heso) {
		for (int i = 0; i <= heso; i++)
			for (int j = 0; j <= heso; j++) {
				if (mark[x + i][y + j])
					return false;
				if ((y - j >= 0) && mark[x + i][y - j])
					return false;
				if ((x - i >= 0) && mark[x - i][y + j])
					return false;
				if ((x - i >= 0) && (y - j >= 0) && mark[x - i][y - j])
					return false;
			}
		return true;
	}

	public void timDiemGiao() {
		int roundX, roundY;
		ToaDo p1, p2, gd;
		for (int i = 0; i < n; i++)
			covered[i] = false;

		for (int i = 0; i < n; i++)
			for (int j = i + 1; j < n; j++) {
				p1 = tapDiem[i];
				p2 = tapDiem[j];
				if (p1.khoangCach(p2) <= 2 * radius) {
					// giao diem thu nhat
					gd = giaoDiem(p1, p2, radius, 1);
					roundX = (int) Math.round(gd.x);
					roundY = (int) Math.round(gd.y);
					if ((roundX >= 0) && (roundY >= 0) && (roundX <= W)
							&& (roundY <= L)
							&& check(roundX, roundY, (int) radius / 20)) {
						tapGiao1.add(new Sensor(gd, radius));
						mark[roundX][roundY] = true;
					}
					// -----------------------------------
					// giao diem thu hai
					gd = giaoDiem(p1, p2, radius, 2);
					roundX = (int) Math.round(gd.x);
					roundY = (int) Math.round(gd.y);
					if ((roundX >= 0) && (roundY >= 0) && (roundX <= W)
							&& (roundY <= L)
							&& check(roundX, roundY, (int) radius / 20)) {
						tapGiao1.add(new Sensor(gd, radius));
						mark[roundX][roundY] = true;
					}
				}
			}
		// ----------------------------------------
		for (int i = 0; i < tapGiao1.size(); i++)
			for (int j = 0; j < n; j++)
				if (tapGiao1.get(i).isCover(tapDiem[j])) {
					tapGiao1.get(i).add(j);
				}
	}

	public static void sapXep() {
		Collections.sort(tapGiao1, new Comparator<Sensor>() {
			public int compare(Sensor s1, Sensor s2) {
				if (s1.getTargetCount() < s2.getTargetCount())
					return 1;
				else {
					if (s1.getTargetCount() == s2.getTargetCount())
						return 0;
					else
						return -1;
				}
			}
		});
	}

	public void baoPhu() throws IOException {
		sapXep();

		int coverMax;
		double totalDisMax;
		Sensor dGiaoMax;
		Integer[] targetList;
		int targetListCount;

		while (true) {
			coverMax = -oo;
			totalDisMax = 0;
			dGiaoMax = new Sensor(new ToaDo(0, 0), radius);
			for (Sensor dGiao : tapGiao1)
				if (dGiao.getTargetCount() > coverMax) {
					coverMax = dGiao.getTargetCount();
					dGiaoMax = dGiao;
				}
			if (coverMax <= 0)
				break;
			fw.write(dGiaoMax.toCircle() + "\n");
			tapSensor.add(new ToaDo(dGiaoMax.center));
			// update lai
			targetList = dGiaoMax.returnList();
			targetListCount = dGiaoMax.getTargetCount();

			for (Sensor dGiao : tapGiao1)
				for (int i = 0; i < targetListCount; i++) {
					dGiao.remove(targetList[i]);
					covered[targetList[i]] = true;
				}
		}

		for (int i = 0; i < n; i++)
			if (!covered[i]) {
				Sensor s = new Sensor(tapDiem[i], radius);
				fw.write(s.toCircle() + "\n");
				tapSensor.add(new ToaDo(s.center));
			}

		relayNum = tapSensor.size();
	}

	public void inDuLieu() throws IOException {

		for (int i=0;i < n;i++)
			fw.write(tapDiem[i].toDecimalString() + "\n");
	}

	public static void sortEdge() {
		Collections.sort(tapEdge, new Comparator<Edge>() {
			public int compare(Edge e1, Edge e2) {
				if (e1.getLength() > e2.getLength())
					return 1;
				else {
					if (e1.getLength() == e2.getLength())
						return 0;
					else
						return -1;
				}
			}
		});
	}

	public static int getRoot(int x) {
		if (root[x] == x)
			return x;
		else
			return (root[x] = getRoot(root[x]));
	}

	public static void addRelay(ToaDo d1, ToaDo d2) throws IOException {
		// them cac relay node noi giua d1 va d2
		double bkR = radius / 2;
		if (d1.khoangCach(d2) <= 2 * bkR)
			return;
		ToaDo d3 = new ToaDo(0, 0);
		double kCach = d1.khoangCach(d2);
		double deltaX = 2 * bkR * Math.abs(d2.getX() - d1.getX()) / kCach;
		double deltaY = 2 * bkR * Math.abs(d2.getY() - d1.getY()) / kCach;
		// -----------------------
		if (d1.getX() < d2.getX())
			d3.setX(d1.getX() + deltaX);
		else
			d3.setX(d1.getX() - deltaX);
		// ------------------------
		if (d1.getY() < d2.getY())
			d3.setY(d1.getY() + deltaY);
		else
			d3.setY(d1.getY() - deltaY);
		// ------------------------------
		tapRelay.add(d3);
		Sensor ss = new Sensor(d3, radius);
		
		fw.write(ss.toCircle() + "\n");
		relayNum++;
		addRelay(d3, d2);
	}

	public void kruskal() throws IOException {
		for (int i = 0; i < tapSensor.size(); i++)
			for (int j = i + 1; j < tapSensor.size(); j++)
				tapEdge.add(new Edge(i, j, tapSensor.get(i).khoangCach(
						tapSensor.get(j))));

		sortEdge();
		root = new int[tapSensor.size() + 2];
		for (int i = 0; i < tapSensor.size(); i++)
			root[i] = i;

		tapAdj = new Adjacent[N_MAX];
		for (int i = 0; i < tapSensor.size(); i++)
			tapAdj[i] = new Adjacent();

		for (Edge e : tapEdge) {
			int v1 = e.id1;
			int v2 = e.id2;
			int p = getRoot(v1);
			int q = getRoot(v2);
			if (p == q)
				continue;
			root[p] = q;
			// ---------------------
			// ----------------------
			tapAdj[v1].addDinhKe(v2);
			tapAdj[v2].addDinhKe(v1);
		}
	}

	public void steiner() throws IOException {
		numIter = 0;
		radius = radius / 2;
		System.out.println("R = " + radius);
		
		for (int i=0;i < tapSensor.size();i++) root[i] = i;

	    while (true) {
	      double maxGain = -1;
	      int uSaved=-1 ,iSaved= -1,vSaved= -1;

	      for (int i=0;i < tapSensor.size();i++)
	      for (int iu=0;iu < tapAdj[i].tapDinhKe.size();iu++)
	      for (int iv=iu+1;iv < tapAdj[i].tapDinhKe.size();iv++) {
	        int u = tapAdj[i].tapDinhKe.get(iu);
	        int v = tapAdj[i].tapDinhKe.get(iv);
	        ToaDo p1 = tapSensor.get(i);
	        ToaDo p2 = tapSensor.get(u);
	        ToaDo p3 = tapSensor.get(v);
	        double gain = ToaDoMethod.gain(p1, p2, p3, radius);
	        if ((gain > 0) && (gain > maxGain) ) {
	          maxGain = gain;
	          iSaved = i;
	          uSaved = u;
	          vSaved = v;
	        }
	      }
	      if (maxGain > 1) {
	    	numIter ++;
	        ToaDo p1 = tapSensor.get(iSaved);
	        ToaDo p2 = tapSensor.get(uSaved);
	        ToaDo p3 = tapSensor.get(vSaved);
	        tapSensor.add(ToaDoMethod.getSteinerPoint(p1, p2, p3));
	        int stp = tapSensor.size() - 1;
	        tapAdj[stp] = new Adjacent();
	        //-------------------------------------------------
			Sensor ss = new Sensor(tapSensor.get(stp), radius);
			fw.write(ss.toCircle() + "\n");
			relayNum++;
	        //------------------------------------------------
	        int ru = getRoot(uSaved);
	        int rv = getRoot(vSaved);
	        int ri = getRoot(iSaved);
	        root[ru] = root[rv] = ri;
	        tapAdj[iSaved].deleteDinhKe(uSaved);
	        tapAdj[iSaved].deleteDinhKe(vSaved);
	        tapAdj[uSaved].deleteDinhKe(iSaved);
	        tapAdj[vSaved].deleteDinhKe(iSaved);
	        //------------------------------------
	        tapAdj[iSaved].addDinhKe(stp);
	        tapAdj[uSaved].addDinhKe(stp);
	        tapAdj[vSaved].addDinhKe(stp);
	        tapAdj[stp].addDinhKe(iSaved);
	        tapAdj[stp].addDinhKe(uSaved);
	        tapAdj[stp].addDinhKe(vSaved);
	      }
	      else break;
	    }
	    
	    radius = radius * 2;
	    for (int i=0;i < tapSensor.size();i++)
	    	for (Integer j: tapAdj[i].tapDinhKe)
	    		if (i < j)
	    			addRelay(tapSensor.get(i), tapSensor.get(j));
	    radius = radius / 2;
	}
	
	public static void xuat() {
		System.out.println("Number iter of Steiner = " + numIter);
		
	}
	
	public boolean checkIsConnected(int[][] Graph, ArrayList<Integer> Redudant) {
		int i = 0;
		int numVertex = relayNum;
		ArrayList<Object> listVS = new ArrayList<>();
		int visit[] = new int[numVertex];
		Stack<Object> stack = new Stack<>();
		listVS.add(i);
		visit[i] = 1;
		stack.push(i);
		while (!stack.empty()) {
			i = (int) stack.peek();
			int count =0;
			for(int j = 0; j < visit.length; j++) {
				if(!Redudant.contains(j)) {
				if((Graph[i][j] >0) && visit[j] != 1) {
					visit[j] = 1;
					listVS.add(j);
					stack.push(j);
					break;
				}else {
					count++;
				}
				}
				else { count++;}
				
			}
			if(count == visit.length) {
				stack.pop();
			}
			
		}
		for(int k = 0; k < visit.length; k++) {
			if(visit[k] != 1 && !Redudant.contains(k))
				return false;
		
		}
		return true;
		
	}
	
	public void redudantRelay() {
		int[][] Graph = new int[relayNum][relayNum];
		for(int i =0; i <relayNum; i ++){
			for(int j =0; j < relayNum; j++) {
				Graph[i][j] = 0;
			}
		}
		ArrayList<ToaDo> allRelay = new ArrayList<>();
		allRelay.addAll(tapSensor);
		allRelay.addAll(tapRelay);
		for(int i =0; i < allRelay.size(); i++) {
			for(int j = i+1; j < allRelay.size(); j++) {
				if(allRelay.get(i).khoangCach(allRelay.get(j)) <= 2*radius+ 0.001) {
					Graph[i][j] = Graph[j][i] = 1;
				}
			}
		}
		
		ArrayList<Integer> Redudant = new ArrayList<>();
		for(int i = 0; i < tapRelay.size(); i++ ) {
			Redudant.add(i);
			if(!checkIsConnected(Graph, Redudant)) {
				Redudant.remove(Redudant.size() -1);
			}
			
		}
		
		allRelay.removeAll(tapRelay);
		ArrayList<ToaDo> tapRemove = new ArrayList<>();
		for (Integer i: Redudant) {
			tapRemove.remove(tapRelay.get(i));
		}
		tapRelay.removeAll(tapRemove);
		allRelay.addAll(tapRelay);
		
		relayNum = allRelay.size();
		
	}
	
	
	
	public static void main(String[] args) throws IOException {
		Locale lc = new Locale("en", "UN");
		Locale.setDefault(lc);
		for(int i=1; i<=18; i++) {
			 System.out.println("Test-Case: " +i);
			 Main x = new Main();
			 String path = "./test2/Testmip/Test";
			 x.nhapDuLieu(path+i+".inp", path+i+".out");
			 long start = System.nanoTime();
			 x.inDuLieu();
			 x.timDiemGiao();
			 x.baoPhu();
		
			 x.kruskal();
			 x.steiner();
			 x.redudantRelay();
		
			 fw.close();
			 long end = System.nanoTime();
			 System.out.println(relayNum + " relay added");
			 System.out.println("Time =: " + (end - start)*1E-9);
			 xuat();
			 System.out.println("----------------------");
	}
}
}