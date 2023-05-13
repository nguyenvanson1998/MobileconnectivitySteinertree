package Main;

import java.math.BigInteger;
import java.util.Scanner;

public class Test1 {

	public static void main(String[] args) {
		BigInteger a,b,c;
		Scanner scanner = new Scanner(System.in);
		a = scanner.nextBigInteger();
		b = scanner.nextBigInteger();
		c = a.add(b);
		System.out.print(c);
		
		
	}

}
