package com.example.repository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository {
    void save(Object user);
}
