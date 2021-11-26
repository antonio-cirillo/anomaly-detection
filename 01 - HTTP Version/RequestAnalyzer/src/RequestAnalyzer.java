import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.net.UnknownHostException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RequestAnalyzer {
    public static void main(String[] args) throws IOException {
        ArrayList<String> urls = new ArrayList<String>();
        BufferedReader csvReader = new BufferedReader(new FileReader("C:\\Users\\adipi\\Documents\\Università\\I anno\\Reti geografiche\\Assignment\\Assigmenti_1\\WebsiteScraper\\index.csv"));
        String row = "";
        int rowCounter = 0;
        while ((row = csvReader.readLine()) != null) {
            rowCounter++;
            if(rowCounter < 141 || rowCounter > 239) continue;
            if(rowCounter % 2 == 0) continue;
            int begin = row.indexOf(">", 20);
            int end = row.indexOf("<", 37);
            row = row.substring(begin + 1, end);
            urls.add(row);
        }
        csvReader.close();

        System.setProperty("webdriver.chrome.driver", "C:\\ProgramData\\chocolatey\\lib\\chromedriver\\tools\\chromedriver.exe");
        WebDriver driver = new ChromeDriver();
        for(String address : urls){
            driver.get("http://www." + address);
        }
        driver.close();

        //HttpUrlConnection supports only HTTP/1.1
        /*
        int index = 0;
        for(String address : urls){
            index++;
            URL url = new URL("http://www." + address);
            HttpURLConnection con = (HttpURLConnection) url.openConnection();
            con.setRequestMethod("GET");
            con.setConnectTimeout(5000);
            con.setReadTimeout(5000);
            int status = 0;
            try {
                status = con.getResponseCode();
                con.disconnect();
            }
            catch(SocketTimeoutException e){
                status = 408;
            }
            catch(UnknownHostException e){
                status = 400;
            }
            Map<String, List<String>> map = con.getHeaderFields();
            String i = map.toString();
            int bIndex = i.indexOf("HTTP") - 1;
            int eIndex = i.indexOf("]", bIndex) + 1;
            try{
                i = i.substring(bIndex, eIndex);
            } catch (StringIndexOutOfBoundsException e){
                Timestamp timestamp = new Timestamp(System.currentTimeMillis());
                System.out.println(index + "\t[" + timestamp.toString() + "]\tURL: " + url.toString() + "\tHttpProtocol: Encrypted");
            }
            Timestamp timestamp = new Timestamp(System.currentTimeMillis());
            System.out.println(index + "\t[" + timestamp.toString() + "]\tURL: " + url.toString() + "\tHttpProtocol: " + i);
        }*/
    }
}
