package Main;

import java.util.ArrayList;
import java.lang.Math;
import java.util.Random;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
public class createTest1 {
	// rand so nguyen trong khoang (a,b)
	static int NoCar;
	static int NoCk;
	static int NoTram;
	static int H, W;
	static int R;
	public static ArrayList<Node> ListNode;
	public createTest1(int NoCar, int NoCk, int W, int H, int R) {
		this.NoCar = NoCar;
		this.NoCk = NoCk;
		this.W = W;
		this.H= H;
		this.R = R;
		ListNode = new ArrayList<Node>();
	}
	public static int randrange (double min, double max) {
		int x;
		return x =(int)( Math.random()*((max-min) +1) + min);
	}
	static void printTest(int H,int W, int R, int NoCar, int NoCk, ArrayList<Node> FS) throws Exception {
    	BufferedWriter out = new BufferedWriter(new FileWriter(new File("./Testmip/Test18.inp")));
    	Random randx = new Random();
    	out.write(W + " " + H + "\r\n");
    	out.write(randx.nextInt(W) + " " + randx.nextInt(H) + "\r\n");
    	out.write(NoCar +"\r\n" );
    	out.write(R+ "\r\n");
    	out.write(NoCk +"\r\n");
        for(Node S: FS)
        	out.write(S.x + " "+ S.y + "\r\n");
        
        out.close();                         
	}
	
	public static void main(String[] args ) throws Exception {
		createTest1 a = new createTest1(5,2,50,50,2);
		Random rand = new Random();
		NoTram = NoCar*NoCk;
		int N = NoTram;
		
		// how to find the GCD overload from N =  a*b; with a-b min
		int subx = 0;
		int suby = 0;
		for(int i =  (int)(Math.sqrt(N)); i>=1; i-- ) {
			if(N%i == 0) {
				subx = i;
				break;
				
			}
		}
		suby = N/subx;
		
		int[] mark = new int[N+1];
		for(int i = 0 ; i < N+1 ; i ++)
			mark[i]  = 0;
		for(int i = 0 ; i < NoCar; i++)
			for(int j = 0 ; j < NoCk; j++) {
				Node aij = new Node(i,j);
				do {
				int m = rand.nextInt(N+1);
				if(m !=0 && mark[m]== 0) {
					aij.area = m;
					mark[m] = 1;
				}
				
				}while(aij.area == 0);
				ListNode.add(aij);
			}
	
		for(Node go: ListNode) {
			System.out.println(go.area);
			go.x =randrange( W/subx*((go.area-1)%subx)+ W/(4*subx), W/subx*((go.area-1)%subx +1)- W/(4*subx));
			go.y =randrange( H/suby*((go.area-1)/subx)+ H/(4*suby), H/suby*((go.area-1)/subx+1)- H/(4*suby));

			
		}
		
		
		printTest(H, W, R, NoCar, NoCk, ListNode);
		System.out.println("Test xong");
		
				}
			
				
		
		
		
		
		
		
	}

