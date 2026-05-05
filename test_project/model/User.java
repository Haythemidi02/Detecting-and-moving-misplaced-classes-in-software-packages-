package com.example.model;
import javax.persistence.Entity;

@Entity
public class User {
    private Long id;
    private String name;
    public Long getId() { return id; }
}
