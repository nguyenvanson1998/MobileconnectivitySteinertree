package Main;
import java.lang.Math;
import java.util.HashMap;
import java.util.Random;

public class Node {
	public double x,y;
	public int xe;
	public int chuki;
	public int area;
	Random rand = new Random();
	public Node(double x, double y) {
		this.x = x;
		this.y = y;
	}
	public Node(int xe, int chuki) {
		this.xe = xe;
		this.chuki = chuki;
	}
	public double d(Node a) {
		return Math.sqrt((x-a.x)*(x-a.x) + (y-a.y)*(y-a.y));
	}
	
	
}
