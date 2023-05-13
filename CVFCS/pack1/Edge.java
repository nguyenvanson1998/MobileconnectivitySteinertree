package pack1;
import java.util.*;
import java.text.*;

public class Edge{
  public int id1,id2;
  public ToaDo point1,point2;
  public double doDai;
  public boolean isXoa;

  public Edge(int a,int b,ToaDo p1,ToaDo p2,double dd){
    id1 = a;
    id2 = b;
    point1 = p1;
    point2 = p2;
    doDai = dd;
    isXoa = false;
  }
}