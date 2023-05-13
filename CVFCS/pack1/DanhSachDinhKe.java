package pack1;
import java.util.*;
import java.text.*;

public class DanhSachDinhKe{
  ArrayList<Integer> adj;
  public DanhSachDinhKe(){
    adj = new ArrayList<Integer>();
  } 
  public void push(int x){
    adj.add(x);
  }
  public int get(int id){
    return adj.get(id);
  }
  public void show(){
    System.out.println(adj);
  }
  public int size(){
    return adj.size();
  }
}