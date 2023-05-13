package pack1;

public class Edge{
  public int id1,id2;
  private double length;
  public Edge(int verA,int verB,double len){
    id1 = verA;
    id2 = verB;
    length = len;
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