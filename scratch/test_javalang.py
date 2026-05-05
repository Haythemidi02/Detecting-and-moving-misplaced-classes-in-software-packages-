import javalang

code = """
package com.example;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class MyService extends BaseService implements IService {
    private String name;
    public void doSomething() {
        System.out.println("Hello");
    }
}
"""

tree = javalang.parse.parse(code)
print(f"Package: {tree.package.name}")
for type_decl in tree.types:
    print(f"Type: {type_decl.name}")
    print(f"Extends: {type_decl.extends.name if type_decl.extends else 'None'}")
    print(f"Implements: {[i.name for i in type_decl.implements]}")
    for method in type_decl.methods:
        print(f"Method: {method.name}")
    for field in type_decl.fields:
        for decl in field.declarators:
            print(f"Field: {decl.name}")
    for annotation in type_decl.annotations:
        print(f"Annotation: {annotation.name}")
