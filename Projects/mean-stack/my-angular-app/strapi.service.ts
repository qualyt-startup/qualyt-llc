import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class StrapiService {
  private baseUrl = 'http://localhost:1337/api'; // Adjust this URL to your Strapi backend

  constructor(private http: HttpClient) {}

  // Fetch data from Strapi (e.g., articles collection)
  getArticles(): Observable<any> {
    return this.http.get(`${this.baseUrl}/articles`);
  }
}
