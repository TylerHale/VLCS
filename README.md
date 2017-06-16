# Build Package
```
$ mvn package
```

# Run example
```
$ java -cp target/vlcs-1.0-SNAPSHOT-jar-with-dependencies.jar VOCLRunner powersupply.arff 13 2
```
Filter out stderr
```
$ java -cp target/vlcs-1.0-SNAPSHOT-jar-with-dependencies.jar VOCLRunner powersupply.arff 13 2 2>/dev/null
```
