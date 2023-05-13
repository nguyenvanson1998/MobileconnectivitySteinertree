package pack1;
import java.util.*;
import java.text.*;

public class Adjacent{
  public List<Integer> tapDinhKe;
  public Adjacent() {
    tapDinhKe = new ArrayList<Integer>();
  }
  public void addDinhKe(int u) {
    tapDinhKe.add(u);
  }
  public void deleteDinhKe(int u) {
    tapDinhKe.remove((Integer) u);
  }
}