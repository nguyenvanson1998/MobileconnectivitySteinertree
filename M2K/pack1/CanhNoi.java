package pack1;
import java.util.*;
import java.text.*;

public class CanhNoi{
  public int id1,id2;
  private double length;
  public boolean isXoa;
  public CanhNoi(int verA,int verB,double len){
    id1 = verA;
    id2 = verB;
    length = len;
    isXoa = false;
  }
  public int getVerA(){
    return id1;
  }
  public int getVerB(){
    return id2;
  }
  public double getLength(){
    return length;
  }
}